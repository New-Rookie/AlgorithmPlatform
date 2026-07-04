from __future__ import annotations

import importlib
import platform
import sys


def safe_import(name: str):
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print()

    torch, torch_err = safe_import("torch")
    if torch_err:
        print("[FAIL] torch import failed")
        print(repr(torch_err))
        return

    print("[OK] torch imported")
    print("torch version:", torch.__version__)
    print("torch cuda version:", getattr(torch.version, "cuda", None))
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print()

    torchvision, tv_err = safe_import("torchvision")
    if tv_err:
        print("[FAIL] torchvision import failed")
        print(repr(tv_err))
        print()
        print("Most likely cause: torch and torchvision are incompatible or installed from different wheel indexes.")
        print("Fix by reinstalling torch + torchvision + torchaudio together from the same PyTorch index URL.")
        return

    print("[OK] torchvision imported")
    print("torchvision version:", torchvision.__version__)
    print()

    try:
        from torchvision.ops import nms

        print("[OK] torchvision.ops.nms is available:", nms)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] torchvision.ops.nms is not available")
        print(repr(exc))
        print()
        print("This is the same root cause as: RuntimeError: operator torchvision::nms does not exist")
        print("Reinstall torch/torchvision/torchaudio as a matched set.")
        return

    transformers, tf_err = safe_import("transformers")
    if tf_err:
        print("[FAIL] transformers import failed")
        print(repr(tf_err))
        return
    print("[OK] transformers imported")
    print("transformers version:", transformers.__version__)

    try:
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        print("[OK] VideoMAE classes imported")
        print(VideoMAEForVideoClassification, VideoMAEImageProcessor)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] VideoMAE import failed")
        print(repr(exc))
        return

    print()
    print("Environment check passed.")


if __name__ == "__main__":
    main()
