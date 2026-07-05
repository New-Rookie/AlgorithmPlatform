from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, get_cosine_schedule_with_warmup

try:
    from torch.optim import AdamW
except ImportError:  # pragma: no cover
    from transformers import AdamW

from algorithm_platform.data.manifest import build_label_maps, load_manifest, save_label_maps
from algorithm_platform.data.video import sample_video_clip


class AssemblyClipDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        processor: VideoMAEImageProcessor,
        label2id: Dict[str, int],
        split: str,
        num_frames: int,
        max_samples: Optional[int] = None,
    ):
        self.df = load_manifest(manifest_path, split=split)
        if max_samples is not None:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42).reset_index(drop=True)
        self.processor = processor
        self.label2id = label2id
        self.num_frames = num_frames
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = sample_video_clip(
            path=str(row.video_path),
            start_sec=float(row.start),
            end_sec=float(row.end),
            num_frames=self.num_frames,
        )
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = self.label2id[str(row.label)]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            losses.append(float(outputs.loss.item()))
            preds = outputs.logits.argmax(dim=-1)
            total += labels.numel()
            correct += int((preds == labels).sum().item())
    model.train()
    return {"loss": float(np.mean(losses)) if losses else None, "accuracy": correct / max(total, 1)}


def _validate_before_training(manifest_path: str, output_dir: str) -> None:
    if not Path(manifest_path).exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run dataset bootstrap first, for example:\n"
            "python scripts/bootstrap_assembly101.py --views v1 --max-recordings 80 --max-rows 12000 --num-labels 6"
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def train(cfg):
    print("=" * 80, flush=True)
    print("AlgorithmPlatform Training Start", flush=True)
    print("=" * 80, flush=True)

    seed = int(cfg.raw.get("seed", 42))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = cfg["paths"]["manifest"]
    output_dir = cfg["paths"]["output_dir"]
    num_frames = int(cfg["model"].get("num_frames", 16))
    _validate_before_training(manifest_path, output_dir)

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Model: {cfg['model']['name']}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Num frames: {num_frames}", flush=True)

    all_df = load_manifest(manifest_path)
    if len(all_df) == 0:
        raise ValueError(f"Manifest has zero rows: {manifest_path}")
    print(f"Total manifest rows: {len(all_df)}", flush=True)
    print(f"Split counts: {all_df['split'].value_counts().to_dict()}", flush=True)
    print(f"Label counts: {all_df['label'].value_counts().head(20).to_dict()}", flush=True)

    label_maps = build_label_maps(all_df)
    if len(label_maps.label2id) < 2:
        raise ValueError(f"Need at least 2 labels to train a classifier. Found: {label_maps.label2id}")
    save_label_maps(label_maps, output_dir)
    print(f"Labels: {label_maps.label2id}", flush=True)

    print("Loading processor and model...", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model"]["name"])
    model = VideoMAEForVideoClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=len(label_maps.label2id),
        label2id=label_maps.label2id,
        id2label={str(k): v for k, v in label_maps.id2label.items()},
        ignore_mismatched_sizes=bool(cfg["model"].get("ignore_mismatched_sizes", True)),
    ).to(device)

    train_split = cfg["data"].get("train_split", "train")
    val_split = cfg["data"].get("val_split", "val")
    train_ds = AssemblyClipDataset(
        manifest_path=manifest_path,
        processor=processor,
        label2id=label_maps.label2id,
        split=train_split,
        num_frames=num_frames,
        max_samples=cfg["data"].get("max_train_samples"),
    )
    val_ds = AssemblyClipDataset(
        manifest_path=manifest_path,
        processor=processor,
        label2id=label_maps.label2id,
        split=val_split,
        num_frames=num_frames,
        max_samples=cfg["data"].get("max_val_samples"),
    )

    print(f"Train rows ({train_split}): {len(train_ds)}", flush=True)
    print(f"Val rows ({val_split}): {len(val_ds)}", flush=True)
    if len(train_ds) == 0:
        raise ValueError(
            f"Training split is empty: {train_split}. Check manifest split column: {manifest_path}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    total_steps = max(1, len(train_loader) * int(cfg["training"]["epochs"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    metrics_path = os.path.join(output_dir, "train_metrics.jsonl")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("mixed_precision", True)) and torch.cuda.is_available())
    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 1))

    print("Starting training loop...", flush=True)
    print(f"Epochs: {cfg['training']['epochs']} | Batches per epoch: {len(train_loader)} | Grad accumulation: {grad_accum}", flush=True)

    global_step = 0
    best_acc = -1.0
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            epoch_losses.append(float(loss.item() * grad_accum))
            pbar.set_postfix(loss=np.mean(epoch_losses[-20:]))

        val_metrics = evaluate(model, val_loader, device) if len(val_ds) else {"loss": None, "accuracy": None}
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "global_step": global_step,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record, flush=True)

        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch + 1}"))
        processor.save_pretrained(os.path.join(output_dir, f"epoch_{epoch + 1}"))
        if val_metrics["accuracy"] is not None and val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training finished. Model saved to: {output_dir}", flush=True)
