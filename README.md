# AlgorithmPlatform

Python assembly action recognition platform.

V1 focuses on recognition only. It trains a VideoMAE action model with an Assembly101 subset and supports single-video inference plus simulated multi-stream inference.

## Main commands

```bash
python scripts/inspect_dataset.py --config configs/dataset_assembly101.yaml
python scripts/prepare_dataset.py --config configs/dataset_assembly101.yaml
python scripts/train.py --config configs/train.yaml
python scripts/infer_single.py --config configs/infer_single.yaml
python scripts/infer_multi.py --config configs/infer_multi.yaml
```

## Install

Install PyTorch separately for your own CPU/CUDA environment. `requirements.txt` intentionally does not include torch.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

See `docs/GETTING_STARTED.md` for the full workflow.

## Dataset policy

Do not download the full Assembly101 dataset first. Start with a medium subset:

```text
recordings: 50-100
action classes: 5-8
clips: 5000-15000
```

Download annotations first, inspect action distribution, then download only the selected recordings/videos.

## Outputs

Training output:

```text
outputs/checkpoints/videomae_assembly_v1/
```

Inference output:

```text
outputs/inference/*.csv
outputs/events/*.jsonl
outputs/clips/
```

## Boundary

This repo validates the training and inference pipeline with public data. It is not a production model for a real factory until real site data is collected, labeled, and used for fine-tuning.
