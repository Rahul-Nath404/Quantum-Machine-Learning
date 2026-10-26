import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from qml_app.config import TrainingConfig
from qml_app.training import train_variational_model


class TrainingCheckpointTests(unittest.TestCase):
    def test_early_stopping_restores_best_validation_weights(self):
        model = nn.Linear(1, 1, bias=False)
        validation_losses = iter([0.1, 0.2])
        epoch = 0

        def run_epoch(model, loader, criterion, optimizer, device, train_mode, grad_clip=None):
            nonlocal epoch
            if train_mode:
                epoch += 1
                with torch.no_grad():
                    model.weight.fill_(epoch)
                return 1.0, 1, 1
            return next(validation_losses), 1, 1

        config = TrainingConfig(
            epochs=3,
            patience=1,
            min_delta=0.0,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "qml_app.training._run_epoch", side_effect=run_epoch
        ):
            train_variational_model(
                model=model,
                dataloaders={"train": [], "val": []},
                config=config,
                artifacts_dir=Path(directory),
            )

        self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
