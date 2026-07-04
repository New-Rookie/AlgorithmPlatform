# Getting Started

This guide is the clean workflow for the current repository version.

The project has one purpose in V1:

```text
Assembly101 subset -> VideoMAE training -> single inference -> simulated multi-stream inference
```

PyTorch is intentionally not listed in `requirements.txt`. Install it separately according to your GPU/CPU environment.

---

## 1. Install

### 1.1 Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 1.2 Install PyTorch separately

CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Install project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 1.4 Login to Hugging Face

Assembly101 is gated. You must accept the dataset conditions on Hugging Face, then login locally:

```bash
huggingface-cli login
```

---

## 2. Download Assembly101 partially

Do not download the full dataset.

Recommended V1 subset:

```text
50-100 recordings
1-2 views first, usually v1 only at the beginning
5000-15000 clips after manifest preparation
```

---

### 2.1 Download annotations only

```bash
huggingface-cli download cvml-nus/assembly101 \
  --repo-type dataset \
  --local-dir data/raw/assembly101 \
  --include "annotations/*" "annotations/**/*"
```

After this, find the real annotation CSV path under:

```text
data/raw/assembly101/annotations/
```

Then set the path in:

```text
configs/dataset_assembly101.yaml
```

Example:

```yaml
paths:
  annotations: data/raw/assembly101/annotations/coarse-annotations.csv
```

The exact filename may differ, so check your downloaded annotations folder.

---

### 2.2 Inspect annotation labels

```bash
python scripts/inspect_dataset.py --config configs/dataset_assembly101.yaml
```

This outputs:

```text
outputs/dataset/annotation_summary.csv
```

Open it and choose 5-8 high-frequency action labels.

Then write them into:

```text
configs/dataset_assembly101.yaml
```

Example:

```yaml
subset:
  labels: [take, place, attach, insert, screw, inspect]
```

The names must match the real labels in `annotation_summary.csv`.

---

### 2.3 Select recording names

```bash
python scripts/select_assembly101_recordings.py \
  --annotations data/raw/assembly101/annotations/coarse-annotations.csv \
  --video-col video_path \
  --label-col action \
  --labels take,place,attach,insert,screw,inspect \
  --max-recordings 80 \
  --output outputs/dataset/selected_recordings.txt
```

Adjust column names and labels according to your annotation file.

This creates:

```text
outputs/dataset/selected_recordings.txt
```

Each line is one recording folder name.

---

### 2.4 Dry-run partial download

Start with one fixed view, for example `v1`, not all 12 views.

```bash
python scripts/download_assembly101_subset.py \
  --recordings-file outputs/dataset/selected_recordings.txt \
  --local-dir data/raw/assembly101 \
  --views v1 \
  --include-annotations \
  --dry-run
```

Check printed patterns. They should look like:

```text
recordings/<recording_name>/C10095_rgb.mp4
```

---

### 2.5 Actually download partial videos

```bash
python scripts/download_assembly101_subset.py \
  --recordings-file outputs/dataset/selected_recordings.txt \
  --local-dir data/raw/assembly101 \
  --views v1 \
  --include-annotations
```

Useful view choices:

```text
v1      -> C10095_rgb.mp4 only
fixed   -> all 8 fixed RGB views, much larger
ego     -> egocentric monochrome views
all     -> fixed + egocentric, not recommended initially
```

For your current purpose, start with:

```text
--views v1
```

Only increase to `v1,v2` or `fixed` after the full training/inference pipeline works.

---

## 3. Prepare manifest

After videos are downloaded, run:

```bash
python scripts/prepare_dataset.py --config configs/dataset_assembly101.yaml
```

Outputs:

```text
data/processed/assembly101/manifest.csv
data/processed/assembly101/label_map.json
```

If it says no rows left, usually one of these is wrong:

```text
annotation path
video column name
video root path
label names
view files not downloaded
```

---

## 4. Train

```bash
python scripts/train.py --config configs/train.yaml
```

Output:

```text
outputs/checkpoints/videomae_assembly_v1/
```

This directory is the model directory used by inference configs.

---

## 5. Single-video inference

Set `configs/infer_single.yaml`:

```yaml
paths:
  model_dir: outputs/checkpoints/videomae_assembly_v1
  video_path: data/raw/assembly101/recordings/<recording_name>/C10095_rgb.mp4
```

Run:

```bash
python scripts/infer_single.py --config configs/infer_single.yaml
```

Outputs:

```text
outputs/inference/single_predictions.csv
outputs/events/single_events.jsonl
outputs/clips/
```

---

## 6. Multi-stream simulated inference

Set `configs/infer_multi.yaml`:

```yaml
paths:
  model_dir: outputs/checkpoints/videomae_assembly_v1

streams:
  - stream_id: stream_001
    video_path: data/raw/assembly101/recordings/<recording_1>/C10095_rgb.mp4
    station_id: station_001
    employee_id: emp_001
    sop_id: sop_assembly101
  - stream_id: stream_002
    video_path: data/raw/assembly101/recordings/<recording_2>/C10095_rgb.mp4
    station_id: station_002
    employee_id: emp_002
    sop_id: sop_assembly101
```

Run:

```bash
python scripts/infer_multi.py --config configs/infer_multi.yaml
```

Output:

```text
outputs/events/multi_events.jsonl
```

---

## 7. Practical rule

Do this first:

```text
annotations + 80 recordings + v1 only
```

Do not download all views or full data until the model training and inference outputs are verified.
