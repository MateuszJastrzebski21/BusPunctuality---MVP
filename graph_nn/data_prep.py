import pandas as pd
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import json
import torch
from torch_geometric.data import Data
from plotting_utils import plot_basic_graph
import wandb

class DataPreparation:
    def __init__(self, data_path: str, wandb_run: wandb.sdk.wandb_run.Run | None = None):
        """
        Initialize the DataPreparation class.

        Args:
            data_path (str): Path to the parquet file containing the data
        """
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.selected_columns = [
            'Linia',
            'Lp przystanku',
            'Rodzaj detekcji',
            'Primary Key',
            'stop_desc',
            'stop_lat',
            'stop_lon',
            'Przystanek numer',
            'Przystanek nazwa',
            'arrival_hour',
            'is_weekday',
            'delay',
            'line_encoded',
            'original_lp_przystanku',
            'arrival_minute',
            'arrival_day',
            'arrival_month',
            'arrival_year',
        ]

        self.numerical_columns = [
            'Linia',
            'Lp przystanku',
            'Rodzaj detekcji',
            'stop_lat',
            'stop_lon',
            'arrival_hour',
            'arrival_minute',
            'line_encoded',
            'arrival_day',
            'arrival_month',
            'arrival_year',
        ]

        self.features = ["arrival_hour", "arrival_minute","arrival_day",
                                      "arrival_month", "arrival_year", "is_weekday",
                                      "lat", "lon", "line_encoded"]
        self.wandb_run = wandb_run if wandb_run else None

    def load_data(self, line_number: str = None) -> pl.DataFrame:
        """
        Load data from parquet file and optionally filter by line number.

        Args:
            line_number (str, optional): Bus line number to filter. Defaults to None.

        Returns:
            pl.DataFrame: Loaded and filtered dataframe
        """
        df = pl.read_parquet(self.data_path)
        if line_number:
            df = df.filter(pl.col("Linia") == line_number)
        return df

    def preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess the data by adding new columns and handling missing values.

        Args:
            df (pl.DataFrame): Input dataframe

        Returns:
            pl.DataFrame: Preprocessed dataframe
        """
        df = df.with_columns([
            pl.col("Rozkladowy czas odjazdu").dt.hour().alias("arrival_hour"),
            pl.col("Rozkladowy czas odjazdu").dt.minute().alias("arrival_minute"),
            pl.col("Rozkladowy czas odjazdu").dt.weekday().alias("is_weekday"),
            pl.col("Rozkladowy czas odjazdu").dt.day().alias("arrival_day"),
            pl.col("Rozkladowy czas odjazdu").dt.month().alias("arrival_month"),
            pl.col("Rozkladowy czas odjazdu").dt.year().alias("arrival_year"),
        ])

        # Print null statistics
        null_stats = df.select(pl.all().is_null().sum()).to_dicts()[0]
        print("Null statistics:", null_stats)

        # Handle missing values
        # print przystanek numer and nazwa where lat lon is null
        # missing_lat_lon = df.filter(pl.col("stop_lat").is_null() | pl.col("stop_lon").is_null())
        # print("Missing lat/lon for stops:")
        # print(missing_lat_lon.select(["Przystanek numer", "Przystanek nazwa", "stop_lat", "stop_lon"]))
        # unique_przystanek_numbers = missing_lat_lon.select("Przystanek numer").unique().to_series().to_list()

        # todo change later
        # for now if there are nulls in any one of a column (for which we observed nulls before), we drop all instances from that trip because it can pollute the graphs
        # trip_ids_with_nulls = df.filter(
        #     pl.col("stop_lat").is_null() | pl.col("stop_lon").is_null() |
        #     pl.col("Rzeczywisty czas przyjazdu").is_null() | pl.col("Rzeczywisty czas odjazdu").is_null()
        # )["Primary Key"].unique()
        #
        # print(f"Before removing trips with nulls, number of records: {df.shape[0]}")
        # df = df.filter(~pl.col("Primary Key").is_in(trip_ids_with_nulls))
        # print(f"After removing trips with nulls, number of records: {df.shape[0]}")

        print(f"Before removing trips with nulls, number of records: {df.shape[0]}")
        df = df.drop_nulls()
        print(f"After removing trips with nulls, number of records: {df.shape[0]}")

        # remove extreme delay values
        trips_with_extreme_delay_values = df.filter(
            (pl.col("delay") < -3000) | (pl.col("delay") > 3000)
        )["Primary Key"].unique()
        print(f"Found {len(trips_with_extreme_delay_values)} trips with extreme delay values.")
        print(f"Before removing trips with extreme delay values, number of records: {df.shape[0]}")
        df = df.filter(~pl.col("Primary Key").is_in(trips_with_extreme_delay_values))
        print(f"After removing trips with extreme delay values, number of records: {df.shape[0]}")

        # Convert categorical variables
        df = df.with_columns([
            pl.col("is_weekday").cast(pl.Int8),
            pl.col("Lp przystanku").alias("original_lp_przystanku"),
            pl.col("Linia").cast(pl.Categorical).cast(pl.UInt32).alias("line_encoded")
        ])

        return df

    def prepare_features_and_target(self, df: pl.DataFrame):
        """
        Prepare features and target variables for model training.

        Args:
            df (pl.DataFrame): Preprocessed dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (target) arrays
        """
        X = df.select(self.selected_columns)
        y = df.select(["delay"]).to_numpy()

        # Scale numerical features
        X[self.numerical_columns] = self.scaler.fit_transform(X[self.numerical_columns])

        # Save scaler for later use
        self.save_scaler()

        return X, y

    def save_scaler(self, path: str = "models/scaler.joblib"):
        """
        Save the fitted scaler to disk.

        Args:
            path (str): Path to save the scaler
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

        artifact = wandb.Artifact('numerical_feature_scaler', type='scaler',
                                  description=f'{self.scaler.__class__}',
                                 )

        artifact.add_file(path)
        self.wandb_run.log_artifact(artifact)

    def load_scaler(self, path: str = "models/scaler.pkl"):
        """
        Load a previously saved scaler.

        Args:
            path (str): Path to the saved scaler
        """
        self.scaler = joblib.load(path)

    def plot_feature_distribution(self, df):
        sns.set(style="whitegrid")

        fig, axs = plt.subplots(9, 2, figsize=(14, 20))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        for i, feature in enumerate(df.columns):
            row, col = divmod(i, 2)
            ax = axs[row, col]
            sns.histplot(data=df, x=feature, ax=ax, fill=True)
            ax.set_title(f"Distribution of {feature}")

        plt.tight_layout()
        plt.savefig("output/feature_distributions.png")
        wandb.log({"feature_distributions": wandb.Image("output/feature_distributions.png")})
        plt.show()

    def prepare_edge_list(self, df: pl.DataFrame, X, stop_mapping_out_file: str = "output/stop_mapping.json"):
        """
                Prepare the edge list for the graph. List of [(source,dest), (source, destination),...] pairs.
                :param stop_mapping_out_file: Path to the output file for stop mapping.
                :return: Tuple[List[Tuple[int, int]], Dict[int, int]]
                """

        edge_list = []

        stop_ids = sorted(df['Przystanek numer'].unique())
        stop_id_map = {sid: i for i, sid in enumerate(stop_ids)}
        # with open(stop_mapping_out_file, "w") as f:
        #     json.dump(stop_id_map, f)
        #todo fix
        # wandb.log({"stop_mapping": wandb.Table(data=stop_id_map.items(), columns=["stop_id", "index"])})

        for _, group in X.sort(['Primary Key', 'Lp przystanku']).group_by('Primary Key'):
            stops = group['Przystanek numer'].to_list()
            for i in range(len(stops) - 1):
                edge_list.append((stops[i], stops[i + 1]))

        edge_list = list(set(edge_list))
        print(f"Number of edges: {len(edge_list)}")
        return edge_list, stop_id_map

    def prepare_edge_index(self, df, X, edge_list: list[tuple[int, int]] | None = None, stop_id_map: dict[int, int] | None = None, stop_mapping_out_file="stop_mapping.json"):
        """
        Prepare the edge index for the graph.
        :param stop_mapping_out_file: Path to the output file for stop mapping.
        :return: torch.Tensor
        """
        if edge_list is None:
            edge_list, stop_id_map = self.prepare_edge_list(df, X, stop_mapping_out_file)

        edge_index = torch.tensor([
            [stop_id_map[src] for src, dst in edge_list],
            [stop_id_map[dst] for src, dst in edge_list]
        ], dtype=torch.long)
        return edge_index

    def plot_graph(self, df, edge_list_ids):
        stop_id_to_stop_name_dict = dict(zip(df['Przystanek numer'], df['Przystanek nazwa']))
        stop_id_to_name_latlon = {
            stop_id: (name, (lat, lon))
            for stop_id, name, lat, lon in zip(
                df['Przystanek numer'],
                df['Przystanek nazwa'],
                df['stop_lat'],
                df['stop_lon']
            )
        }

        fig = plot_basic_graph(edge_list_ids=edge_list_ids,
                         stop_id_to_name_latlon=stop_id_to_name_latlon)

        wandb.log({"bus_graph_map": wandb.Plotly(fig)})

    def prepare_graph_features(self, df, stop_id_map, edge_index):
        """
        Prepare graph features for each trip.
        :param df: DataFrame containing the data.
        :param stop_id_map: Mapping of stop IDs to indices.
        :param edge_index: Edge index tensor.
        :return: List of Data objects.
        """
        data_list = []

        df = df.sort(['scheduled_trip_start'])
        grouped = df.group_by("Primary Key", maintain_order=True)
        # log a df with two columns: trip id and records per trip id
        trip_counts = grouped.agg(pl.count()).to_pandas()
        wandb.log({
            "trip_counts": wandb.Table(
                data=trip_counts,
                columns=["trip_id", "records_per_trip"]
            )
        })
        #make boxplot of records per trip id


        wandb.log({
            "line_names": wandb.Table(
                data=df.select(["Linia", "line_encoded"]).unique().to_pandas(),
                columns=["line_name", "line_encoded"]
            )
        })

        for trip_id, group_df in grouped:
            # print(f"Processing trip {trip_id} with {len(group_df)} records.")
            group_dict = group_df.to_dict(as_series=False)

            num_nodes = len(stop_id_map)
            x = torch.zeros(num_nodes, 9, dtype=torch.float32)
            y = torch.full((num_nodes,), float('nan'), dtype=torch.float32)


            for i in range(len(group_dict["Przystanek numer"])):
                stop = group_dict["Przystanek numer"][i]
                idx = stop_id_map.get(stop)
                if idx is None:
                    continue  # skip unknown stops


                x[idx] = torch.tensor([
                    group_dict["arrival_hour"][i],
                    group_dict["arrival_minute"][i],
                    group_dict["arrival_day"][i],
                    group_dict["arrival_month"][i],
                    group_dict["arrival_year"][i],
                    group_dict["is_weekday"][i],
                    group_dict["stop_lat"][i],
                    group_dict["stop_lon"][i],
                    group_dict["line_encoded"][i],
                ], dtype=torch.float32)

                y[idx] = group_dict["delay"][i]

            trip_start = group_dict["scheduled_trip_start"][0]
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        #todo check if this is still needed
        # # needs to be sorted by scheduled_trip_start to properly split the data for train/val/test
        # for date in sorted(date_to_data.keys()):
        #     data_list.append(date_to_data[date])

        print(f"Number of graphs: {len(data_list)}")

        return data_list

    def split_data(self, data_list: List[Data], train_size: float = 0.7, val_size: float = 0.15) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split the data into training, validation, and test sets.

        Args:
            data_list (List[Data]): List of Data objects, sorted by scheduled trip time.
            train_size (float): Proportion of data to use for training.
            val_size (float): Proportion of data to use for validation.

        Returns:
            Tuple[List[Data], List[Data], List[Data]]: Training, validation, and test datasets.
        """
        total_size = len(data_list)
        train_end = int(total_size * train_size)
        val_end = int(total_size * (train_size + val_size))

        #todo change for rolling window split

        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
        self.wandb_run.log({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data)
        })

        return train_data, val_data, test_data

    def _extract_features(self, data_split, split_name):
        rows = []
        for data in data_split:
            x = data.x
            y = data.y
            mask = ~torch.isnan(y)
            for row, target in zip(x[mask], y[mask]):
                rows.append(row.tolist() + [target.item()])
        df = pd.DataFrame(rows, columns=self.features + ["delay"])
        df["split"] = split_name
        return df

    def visualize_features_in_train_val_test_splits(self, train_data: List[Data], val_data: List[Data], test_data: List[Data]):
        #todo add dates
        df_train = self._extract_features(train_data, "Train")
        df_val = self._extract_features(val_data, "Validation")
        df_test = self._extract_features(test_data, "Test")

        # Combine for plotting
        df_plot = pd.concat([df_train, df_val, df_test], ignore_index=True)

        # Plot using Seaborn
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(5, 2, figsize=(14, 12))
        self.features += ["delay"]
        for i, feature in enumerate(self.features):
            row, col = divmod(i, 2)
            ax = axs[row, col]
            sns.kdeplot(data=df_plot, x=feature, hue="split", ax=ax, fill=True)
            ax.set_title(f"Distribution of {feature}")

        plt.tight_layout()
        #log file to wandb
        plt.savefig("feature_distributions.png")
        wandb.log({"feature_distributions": wandb.Image("feature_distributions.png")})

        # plt.show()

