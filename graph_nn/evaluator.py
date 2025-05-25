import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
import polars as pl

class Evaluator:
    def __init__(self, save_path: str = "results", wandb_run: wandb.sdk.wandb_run.Run | None = None):
        """
        Initialize the evaluator.

        Args:
            save_path (str): Path to save evaluation results
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

    def calculate_metrics(
            self,
            y_true,
            y_pred
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values

        Returns:
            Dict[str, float]: Dictionary of metrics
        """

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

        self.wandb_run.log(metrics)

        return metrics

    def plot_predictions(
            self,
            y_true,
            y_pred,
            title: str = "Predicted vs Actual Delays",
            save_name: Optional[str] = "predictions_vs_actuals_plot_test"
    ):
        """
        Create scatter plot of predicted vs actual values.

        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values
            title (str): Plot title
            save_name (str, optional): Name to save the plot
        """
        plt.figure(figsize=(10, 8))


        # Create scatter plot
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)

        # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel("Actual Delay")
        plt.ylabel("Predicted Delay")
        plt.title(title)
        plt.legend()

        if save_name:
            plt.savefig(f"output/{save_name}.png")
            plt.close()
            self.wandb_run.log({f"output/{save_name}_plot": wandb.Image(f"output/{save_name}.png")})
        else:
            plt.show()

    def plot_error_distribution(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            title: str = "Prediction Error Distribution",
            save_name: Optional[str] = "error_distribution_plot_test"
    ):
        """
        Plot distribution of prediction errors.

        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values
            title (str): Plot title
            save_name (str, optional): Name to save the plot
        """
        plt.figure(figsize=(10, 6))

        # Calculate errors

        errors = (np.array(y_pred) - np.array(y_true))

        # Create distribution plot
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')

        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()

        plt.savefig(f"output/{save_name}.png")
        self.wandb_run.log({f"{save_name}_plot": wandb.Image(f"output/{save_name}.png")})
        plt.close()


    def plot_training_history(
            self,
            history: Dict[str, list],
            save_name: Optional[str] = None
    ):
        """
        Plot training and validation metrics over epochs.

        Args:
            history (Dict[str, list]): Training history
            save_name (str, optional): Name to save the plot
        """
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot RMSE
        plt.subplot(1, 2, 2)
        plt.plot(history['train_rmse'], label='Train RMSE')
        plt.plot(history['val_rmse'], label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()

        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png")
            self.wandb_run.log({f"{save_name}_plot": wandb.Image(f"{save_name}.png")})
            plt.close()
        else:
            plt.show()

    def plot_per_feature_rmse(
            self,
            feature_stats: pl.dataframe,
            save_name: Optional[str] = "output/metrics_per_feature_test"
    ):
        # # Print per-feature MAE and RMSE
        # print("\n🔎 Per-feature error metrics:")
        # for feature_val, group in sorted(feature_stats.items()):
        #     y_true = np.array(group["targets"])
        #     y_pred = np.array(group["preds"])
        #
        #     mae = mean_absolute_error(y_true, y_pred)
        #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        #
        #     print(f"{feature_val:>10s} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        """
            Plot MAE and RMSE per feature value.
            :param feature_stats: dict from predict_test with structure:
                                  { "hour=6": {"targets": [...], "preds": [...]}, ... }
            """
        # rows = []
        # for feature_val, group in feature_stats.items():
        #     y_true = group["targets"]
        #     y_pred = group["preds"]
        #     mae = mean_absolute_error(y_true, y_pred)
        #     rmse = root_mean_squared_error(y_true, y_pred)
        #     rows.append({"feature": feature_val, "MAE": mae, "RMSE": rmse})
        #
        # df_metrics = pl.DataFrame(rows)
        # print(df_metrics)
        #
        # # split feature column to feature type and value by = in polars
        # # Note: polars does not support str.extract like pandas, so we use str.split
        # df_metrics = df_metrics.with_columns(
        #     pl.col("feature").str.split("=").arr.get(0).alias("feature_type"),
        #     pl.col("feature").str.split("=").arr.get(1).alias("feature_value")
        # )
        #
        # # # Optional: split feature type for subplots
        # # df_metrics["feature_type"] = df_metrics["feature"].str.extract(r"^(line|hour)")
        # # df_metrics["feature_value"] = df_metrics["feature"].str.extract(r"=(\d+)")
        #
        # print(df_metrics)
        # # group by feature type
        # grouped_by_feature_type = df_metrics.group_by("feature_type")
        # for feature_type, group in grouped_by_feature_type:
        #     print(f"\nFeature Type: {feature_type}")
        #     print(group)
        #     # Plot MAE
        #     plt.figure(figsize=(12, 6))
        #     sns.barplot(data=group.sort("feature_value"), x="feature_value", y="MAE")
        #     plt.title("MAE per Feature Value")
        #     plt.xlabel("Feature Value")
        #     plt.ylabel("Mean Absolute Error")
        #     plt.legend(title="Feature Type")
        #     plt.tight_layout()
        #     plt.savefig(f"{save_name}_{feature_type}_mae.png")
        #     self.wandb_run.log({f"{save_name}_{feature_type}_mae_plot": wandb.Image(f"{save_name}_{feature_type}_mae.png")})
        #
        #
        #     # Plot RMSE
        #     plt.figure(figsize=(12, 6))
        #     sns.barplot(data=group.sort_values("feature_value"), x="feature_value", y="RMSE", hue="feature_type")
        #     plt.title("RMSE per Feature Value")
        #     plt.xlabel("Feature Value")
        #     plt.ylabel("Root Mean Squared Error")
        #     plt.legend(title="Feature Type")
        #     plt.tight_layout()
        #     plt.savefig(f"{save_name}_{feature_type}_rmse_plot.png")
        #     self.wandb_run.log({f"{save_name}_{feature_type}_rmse_plot": wandb.Image(f"{save_name}_{feature_type}_rmse_plot.png")})

        # feature_stats is a df  final_df = pl.DataFrame({
        #             "arrival_hour": all_arrival_hours,
        #             "line_encoded": all_lines,
        #             "targets": all_targets,
        #             "preds": all_preds
        #         })
        # plot per feature RMSE
        print(feature_stats)
        for feature in feature_stats.columns:
            if feature not in ["targets", "preds"]:
                # Calculate RMSE per feature value
                print(feature_stats.select(
                    pl.col(feature),
                    pl.col("targets"),
                    pl.col("preds")
                ).describe())
                # rmse_per_value = feature_stats.groupby(feature).agg(
                #     pl.col("targets").alias("targets"),
                #     pl.col("preds").alias("preds")
                # ).with_columns(
                #     (pl.col("preds") - pl.col("targets")).pow(2).sqrt().alias("RMSE")
                # )
                #
                # # Plot RMSE
                # plt.figure(figsize=(12, 6))
                # sns.barplot(data=rmse_per_value, x=feature, y="RMSE")
                # plt.title(f"RMSE per {feature}")
                # plt.xlabel(feature)
                # plt.ylabel("Root Mean Squared Error")
                # plt.tight_layout()
                # plt.savefig(f"{save_name}_{feature}_rmse_plot.png")
                # self.wandb_run.log({f"{save_name}_{feature}_rmse_plot": wandb.Image(f"{save_name}_{feature}_rmse_plot.png")})
                # plt.close()


