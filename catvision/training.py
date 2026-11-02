"""Small-batch classification training with explicit failure handling."""
import math
import os

from ultralytics.models.yolo.classify import ClassificationTrainer

from .run_control import gpu_checkpoint


class LowLoadClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset_path, batch_size=8, rank=0, mode="train"):
        # The upstream trainer doubles validation batches by default.
        return super().get_dataloader(dataset_path, min(batch_size, 8), rank, mode)

    def _handle_nan_recovery(self, epoch):
        loss_ok = self.loss is None or bool(self.loss.isfinite().all())
        fitness_ok = self.fitness is None or math.isfinite(float(self.fitness))
        if not loss_ok or not fitness_ok:
            raise RuntimeError("训练出现NaN/Inf，已停止，禁止自动恢复或重启")
        return False


def disable_automatic_oom_retry(trainer):
    # Version-pinned Ultralytics catches first-epoch OOM and retries up to 3 times.
    # Mark the budget exhausted before every batch so that errors propagate instead.
    trainer._oom_retries = 3


def training_batch_pause(trainer):
    gpu_checkpoint(pause_seconds=0.05)
    if trainer.loss is not None and not bool(trainer.loss.isfinite().all()):
        raise RuntimeError("训练损失出现NaN/Inf，已停止，不自动重试")


def validation_batch_pause(validator):
    gpu_checkpoint(pause_seconds=0.05)


def durable_checkpoints(trainer):
    for path in [trainer.last, trainer.best]:
        if path.is_file():
            with path.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
