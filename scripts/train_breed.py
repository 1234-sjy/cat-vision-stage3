from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask, require_training_enabled


def main():
    require_training_enabled()
    from ultralytics import YOLO
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--name", default="yolo26s_breeds")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA PyTorch")
    config = {
        "data": str(ROOT / "data/classification"), "epochs": args.epochs,
        "imgsz": 320, "batch": 1, "workers": 0, "device": 0,
        "optimizer": "AdamW", "lr0": 0.0008, "lrf": 0.05, "cos_lr": True,
        "patience": 10, "seed": 314159, "deterministic": True,
        "amp": True, "plots": True, "save": True, "erasing": 0.15,
        "project": str(ROOT / "artifacts/training"), "name": args.name,
    }
    model = YOLO(str(ROOT / "weights/yolo26s-cls.pt"))
    started = time.perf_counter()
    model.train(**config)
    best = Path(model.trainer.best)
    destination = ROOT / "weights/breed-best.pt"
    shutil.copy2(best, destination)
    # Store portable paths in the recorded configuration.
    config["data"] = "data/classification"
    config["project"] = "artifacts/training"
    save_json(ROOT / "artifacts/training/final-selection.json", {
        "architecture": "YOLO26s-cls", "initial_weights": "weights/yolo26s-cls.pt",
        "selected_weights": "weights/breed-best.pt", "rule": "best validation classification fitness",
        "training_seconds": time.perf_counter() - started, "config": config,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    })
    print("TRAINING FINISHED", destination, flush=True)


if __name__ == "__main__":
    require_training_enabled()
    with ModelTask("formal-training"):
        main()
