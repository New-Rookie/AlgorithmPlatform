# Environment Guide

`requirements.txt` intentionally does not install PyTorch. Install `torch`, `torchvision`, and `torchaudio` together as a matched set.

Before training, run:

```bash
python scripts/check_env.py
```

---

## Common error

```text
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'VideoMAEForVideoClassification'
```

This usually means `torch` and `torchvision` are incompatible, or one package was installed from PyPI while the other was installed from a PyTorch CUDA/CPU wheel index.

---

## Clean reinstall

### Step 1: remove old packages

```bash
pip uninstall -y torch torchvision torchaudio
```

Optional:

```bash
pip cache purge
```

### Step 2: install a matched set

Choose one.

CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: verify

```bash
python scripts/check_env.py
```

Only train after this script prints:

```text
Environment check passed.
```

---

## Rule

Do not run commands like this after installing CUDA PyTorch:

```bash
pip install torchvision
```

That may replace the matched torchvision wheel and break compiled operators such as NMS.
