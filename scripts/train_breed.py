from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask, require_training_enabled, set_low_load_threads


def main():
    require_training_enabled()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_low_load_threads()
    from ultralytics import YOLO
    import torch
    from catvision.training import (LowLoadClassificationTrainer, disable_automatic_oom_retry,
        training_batch_pause, validation_batch_pause, durable_checkpoints)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--name", default="yolo26s_breeds")
    parser.add_argument("--resume", type=Path, help="Explicitly resume an inspected local checkpoint")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA PyTorch")
    config = {
        "data": str(ROOT / "data/classification"), "epochs": args.epochs,
        "imgsz": 320, "batch": 8, "workers": 0, "device": 0,
        "optimizer": "AdamW", "lr0": 0.0008, "lrf": 0.05, "cos_lr": True,
        "patience": 10, "seed": 314159, "deterministic": True,
        "amp": False, "plots": True, "save": True, "save_period": 5, "erasing": 0.15,
        "cache": False, "scale": 0.35, "dropout": 0.1,
        "project": str(ROOT / "artifacts/training"), "name": args.name,
    }
    prior_recorded_seconds = 0.0
    if args.resume:
        results = args.resume.resolve().parents[1] / "results.csv"
        if results.exists():
            with results.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                prior_recorded_seconds = float(rows[-1]["time"])
    model = YOLO(str(args.resume or ROOT / "weights/yolo26s-cls.pt"))
    model.add_callback("on_train_batch_start", disable_automatic_oom_retry)
    model.add_callback("on_train_batch_end", training_batch_pause)
    model.add_callback("on_val_batch_end", validation_batch_pause)
    model.add_callback("on_model_save", durable_checkpoints)
    started = time.perf_counter()
    model.train(trainer=LowLoadClassificationTrainer, resume=bool(args.resume), **config)
    best = Path(model.trainer.best)
    destination = ROOT / "weights/breed-best.pt"
    shutil.copy2(best, destination)
    with destination.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    # Store portable paths in the recorded configuration.
    config["data"] = "data/classification"
    config["project"] = "artifacts/training"
    save_json(ROOT / "artifacts/training/final-selection.json", {
        "architecture": "YOLO26s-cls", "initial_weights": "weights/yolo26s-cls.pt",
        "selected_weights": "weights/breed-best.pt", "rule": "best validation classification fitness",
        "training_seconds": prior_recorded_seconds + time.perf_counter() - started, "config": config,
        "resumed_from": args.resume.relative_to(ROOT).as_posix() if args.resume and args.resume.is_absolute()
                        else str(args.resume) if args.resume else None,
        "interrupted_partial_batches_excluded_from_training_seconds": bool(args.resume),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "actual_validation_batch_max": 8, "pause_seconds_per_batch": 0.05,
        "automatic_oom_retry": False, "automatic_nan_recovery": False,
    })
    print("TRAINING FINISHED", destination, flush=True)


if __name__ == "__main__":
    require_training_enabled()
    with ModelTask("formal-training"):
        main()
