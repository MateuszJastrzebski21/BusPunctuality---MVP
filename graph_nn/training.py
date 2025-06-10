import os

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
import wandb
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import polars as pl

from model import EarlyStopping


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            wandb_run: wandb.sdk.wandb_run.Run | None = None,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            train_data: list[Data] | None = None,
            val_data: list[Data] | None = None,
            test_data: list[Data] | None = None
    ):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train
            device (str): Device to use for training
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
            use_wandb (bool): Whether to use Weights & Biases for logging
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        wandb.log(
            {"learning_rate": learning_rate, "weight_decay": weight_decay, "optimizer": "Adam", "loss_function": "MSELoss"}
        )

        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
        assert self.train_loader is not None, "train_loader is None"
        self.val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        assert self.val_loader is not None, "val_loader is None"
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        assert self.test_loader is not None, "test_loader is None"

        self.wandb_run = wandb_run


    def train(
            self,
            num_epochs: int = 50,
            model_save_path: str = "models"
    ) -> None:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_losses = []
        val_losses = []

        early_stopping = EarlyStopping(patience=5, delta=0.01)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                mask = ~torch.isnan(batch.y)
                loss = self.criterion(out[mask].squeeze(), batch.y[mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.val_loader:
                    out = self.model(batch.x, batch.edge_index)
                    mask = ~torch.isnan(batch.y)
                    loss = self.criterion(out[mask].squeeze(), batch.y[mask])
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(self.val_loader)
                val_losses.append(avg_val_loss)

            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {early_stopping.best_score:.4f}")
                break

        torch.save(self.model.state_dict(), "output/model_weights.pkl")
        artifact = wandb.Artifact("bus_delay_model", type="model")
        artifact.add_file("output/model_weights.pkl")
        self.wandb_run.log_artifact(artifact)



    def predict_test(self):


        """
           Evaluate the model on the test set, log per-feature metrics, and return predictions/targets.
           """

        self.model.eval()
        all_targets = []
        all_preds = []
        all_lines = []
        all_arrival_hours = []
        all_line_types = []
        weekdays = []


        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)

                mask = ~torch.isnan(batch.y)
                preds = out[mask].squeeze()
                targets = batch.y[mask]

                # Collect the features to group by
                arrival_hours = batch.x[mask, 0].cpu().tolist()  # arrival_hour
                line_encoded = batch.x[mask, 8].cpu().tolist()  # line_encoded
                line_type = batch.x[mask, 9].cpu().tolist()  # line_type
                is_weekend = batch.x[mask, 5].cpu().tolist()  # is_weekend
                # create a df with arrival_hours, line_encoded, prediction, target
                # Create a DataFrame for easier handling
                all_arrival_hours.extend(arrival_hours)
                all_lines.extend(line_encoded)
                all_line_types.extend(line_type)
                weekdays.extend(is_weekend)

                # Also track overall
                all_targets.extend(targets.tolist())
                all_preds.extend(preds.tolist())


        final_df = pl.DataFrame({
            "arrival_hour": all_arrival_hours,
            "line_encoded": all_lines,
            "line_type": all_line_types,
            "weekday": weekdays,
            "targets": all_targets,
            "preds": all_preds
        })

        return final_df

    def predict_with_debug(self, data_loader: DataLoader):
        self.model.eval()
        all_targets = []
        all_preds = []
        all_lines = []
        all_arrival_hours = []
        all_line_types = []
        weekdays = []


        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)

                mask = ~torch.isnan(batch.y)
                preds = out[mask].squeeze()
                targets = batch.y[mask]

                # Collect the features to group by
                arrival_hours = batch.x[mask, 0].cpu().tolist()  # arrival_hour
                line_encoded = batch.x[mask, 8].cpu().tolist()  # line_encoded
                line_type = batch.x[mask, 9].cpu().tolist()  # line_type
                is_weekend = batch.x[mask, 5].cpu().tolist()  # is_weekend
                # create a df with arrival_hours, line_encoded, prediction, target
                # Create a DataFrame for easier handling
                all_arrival_hours.extend(arrival_hours)
                all_lines.extend(line_encoded)
                all_line_types.extend(line_type)
                weekdays.extend(is_weekend)

                # Also track overall
                all_targets.extend(targets.tolist())
                all_preds.extend(preds.tolist())


        final_df = pl.DataFrame({
            "arrival_hour": all_arrival_hours,
            "line_encoded": all_lines,
            "line_type": all_line_types,
            "weekday": weekdays,
            "targets": all_targets,
            "preds": all_preds
        })

        return final_df




    def predict(self, data_loader: DataLoader):
        pass



