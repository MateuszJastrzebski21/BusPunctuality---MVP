import torch

import wandb
from data_prep import DataPreparation
from graph_nn.evaluator import Evaluator
from training import Trainer
from model import GNN


def init_wandb(project: str = "bus-delay-prediction"):
    """
    Initialize Weights & Biases logging.

    Args:
        project (str): Name of the W&B project
    """
    return wandb.init(
        project=project
    )

run = init_wandb()
print("Starting data loading...")
data_prep_obj = DataPreparation(data_path="combined_cleaned_with_stops_and_delay_trip_start.parquet", wandb_run=run)
print("DataPreparation object created successfully.")
df = data_prep_obj.load_data(line_number="126")[:500]
print(f"Data loaded successfully., shape: {df.shape}")
df = data_prep_obj.preprocess_data(df)
print(f"Data preprocessed successfully., shape: {df.shape}")

X, y = data_prep_obj.prepare_features_and_target(df)
print(f"Features and target prepared successfully., X shape: {X.shape}, y shape: {y.shape}")


edge_list, stop_id_map = data_prep_obj.prepare_edge_list(df, X)
print(f"Edge list and stop ID map prepared successfully., edge_list shape: {len(edge_list)}, stop_id_map size: {len(stop_id_map)}")
edge_index = data_prep_obj.prepare_edge_index(df, X, edge_list, stop_id_map)
print(f"Edge index prepared successfully., edge_index shape: {edge_index.shape}")
data_prep_obj.plot_graph(df, edge_list_ids=edge_list)

data_list = data_prep_obj.prepare_graph_features(df, stop_id_map, edge_index)
print(f"Graph features prepared successfully., data_list length: {len(data_list)}")

train_data, val_data, test_data = data_prep_obj.split_data(data_list)
print(f"Data split into train, validation, and test sets., train_data length: {len(train_data)}, val_data length: {len(val_data)}, test_data length: {len(test_data)}")

data_prep_obj.visualize_features_in_train_val_test_splits(train_data, val_data, test_data)

model = GNN(in_channels=train_data[0].x.shape[1], hidden_channels=32, out_channels=1)

trainer_obj = Trainer(model=model, train_data=train_data, val_data=val_data, test_data=test_data)
trainer_obj.train()
print("Training completed successfully.")
torch.save(model.state_dict(), "output/model_weights.pkl")
artifact = wandb.Artifact("bus_delay_model", type="model")
artifact.add_file("output/model_weights.pkl")
run.log_artifact(artifact)


all_targets, all_preds, feature_stats = trainer_obj.predict_test()
print(feature_stats)

evaluator_obj = Evaluator(wandb_run=run)
evaluator_obj.calculate_metrics(
    y_true=all_targets,
    y_pred=all_preds,
)
evaluator_obj.plot_predictions(
    y_true=all_targets,
    y_pred=all_preds,


)
evaluator_obj.plot_error_distribution(
    y_true=all_targets,
    y_pred=all_preds,
)

evaluator_obj.plot_per_feature_rmse(
    feature_stats=feature_stats,
)


run.finish()

