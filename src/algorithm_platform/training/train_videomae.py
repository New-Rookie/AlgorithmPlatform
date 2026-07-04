import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AdamW
import cv2


class AssemblyDataset(Dataset):
    def __init__(self, manifest_path, processor, split="train", max_samples=None):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split]
        if max_samples:
            self.df = self.df.sample(max_samples)
        self.processor = processor

        self.labels = sorted(self.df["label"].unique())
        self.label2id = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def _load_video(self, path, start, end, num_frames=16):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_f = int(start * fps)
        end_f = int(end * fps)

        frames = []
        total = max(end_f - start_f, 1)
        step = max(total // num_frames, 1)

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f + i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = self._load_video(row.video_path, row.start, row.end)
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = self.label2id[row.label]
        inputs["labels"] = torch.tensor(label)
        return inputs


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = VideoMAEImageProcessor.from_pretrained(cfg["model"]["name"])
    model = VideoMAEForVideoClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=8,
        ignore_mismatched_sizes=True
    ).to(device)

    dataset = AssemblyDataset(cfg["paths"]["manifest"], processor)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    optim = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            if i % cfg["training"]["log_every"] == 0:
                print(f"Epoch {epoch} Step {i} Loss {loss.item()}")

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["paths"]["output_dir"])
    processor.save_pretrained(cfg["paths"]["output_dir"])
