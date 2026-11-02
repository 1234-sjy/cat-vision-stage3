import unittest

import torch

from catvision.training import LowLoadClassificationTrainer


class TrainingFailureTests(unittest.TestCase):
    def test_nan_loss_aborts_without_checkpoint_recovery(self):
        trainer = object.__new__(LowLoadClassificationTrainer)
        trainer.loss = torch.tensor(float("nan"))
        trainer.fitness = 0.8
        with self.assertRaisesRegex(RuntimeError, "自动恢复"):
            trainer._handle_nan_recovery(4)

    def test_finite_training_does_not_request_recovery(self):
        trainer = object.__new__(LowLoadClassificationTrainer)
        trainer.loss = torch.tensor(0.4)
        trainer.fitness = 0.8
        self.assertFalse(trainer._handle_nan_recovery(4))

    def test_nan_validation_fitness_aborts_without_recovery(self):
        trainer = object.__new__(LowLoadClassificationTrainer)
        trainer.loss = torch.tensor(0.4)
        trainer.fitness = float("nan")
        with self.assertRaisesRegex(RuntimeError, "自动恢复"):
            trainer._handle_nan_recovery(4)


if __name__ == "__main__":
    unittest.main()
