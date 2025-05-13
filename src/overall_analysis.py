"""Functions used in 'main.py'."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils import utils

# Set default fontsize for labels and ticks
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


class OverallAnalysis:
    """Generate analysis for all tested local LLM.

    - Combine metrics csv for all tested local LLM.
    - Reduce ratings and generate metrics.
    - Plot confusion matrix for reduced ratings.
    - Plot top N local LLM by selected metric.

    Usage:
        >>> overall = OverallAnalysis()
        >>> df_overall, df_reduced = overall.run()

    Args:
        data_dir (str):
            Relative path to data folder (Default: "./data").
        ratings (list[str]):
            Reduced ratings (Default: ["negative", "neutral", "positive"]).
        top_n (int):
            Top number of local LLM with highest selected metric.
        metric (str):
            Performance metrics either 'f1_score', 'precision', or 'recall'
            (Default: "f1_score").

    Attributes:
        data_dir (str):
            Relative path to data folder (Default: "./data").
        overall_dir (str):
            Relative path to folder to save overall analysis results
            (Default: "./data").
        ratings (list[str]):
            Reduced ratings (Default: ["negative", "neutral", "positive"]).
        top_n (int):
            Top number of local LLM with highest selected metric.
        metric (str):
            Performance metrics either 'f1_score', 'precision', or 'recall'
            (Default: "f1_score").
        overall_path (str):
            Relative path to csv file containing metrics for all
            tested local LLM.
        reduced_path (str):
            Relative path to csv file containing metrics for all
            tested local LLM with reduced ratings.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        ratings: list[str] = ["negative", "neutral", "positive"],
        top_n: int = 10,
        metric: str = "f1_score",
    ) -> None:
        self.data_dir = data_dir
        self.overall_dir = f"{data_dir}/overall"
        self.ratings = ratings
        self.top_n = top_n
        self.metric = metric
        self.overall_path = f"{self.overall_dir}/overall.csv"
        self.reduced_path = f"{self.overall_dir}/reduced.csv"

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate overall metrics and metrics for reduced ratings."""

        # Combine metrics csv for all tested local LLM
        df_overall = self.gen_overall()

        # Reduce ratings and generate metrics
        df_reduced = self.reduce_ratings()

        # Plot top N local LLM by selected metric
        self.plot_top_n(df_overall, is_reduced=False)
        self.plot_top_n(df_reduced, is_reduced=True)

        return df_overall, df_reduced

    def gen_overall(self) -> pd.DataFrame:
        """Combine metrics csv files for various local LLM models."""

        df_list = []
        models_list = []
        family_list = []

        for fpath in Path(self.data_dir).rglob("metrics*.csv"):
            # Load metrics csv files as DataFrame
            df_list.append(pd.read_csv(fpath))
            models_list.append(fpath.stem.split("_", maxsplit=1)[1])
            family_list.append(fpath.parent.name)

        # Concatentate DataFrames row-wise
        df = pd.concat(df_list, axis=0).reset_index(drop=True)

        # Append 'models' column
        df.insert(0, "model", models_list)
        df.insert(0, "family", family_list)

        # Save as csv file
        utils.create_folder(self.overall_dir)
        df.to_csv(self.overall_path, index=False)

        return df

    def reduce_ratings(self) -> pd.DataFrame:
        """Reduce ratings scale (1 to 5) to (1 to 3)."""

        metrics_list = []

        for fpath in Path(self.data_dir).rglob("ratings*.csv"):
            # Load ratings csv files as DataFrame
            df_ratings = pd.read_csv(fpath)

            # Convert ratings from 0 to 5 to 'negative', 'neutral' and 'positive'
            df_ratings = self.convert_ratings(df_ratings)

            # Compute f1 score, precision, recall and confusion matrix
            metrics_list.append(self.gen_metrics(df_ratings, fpath))

        # Convert to DataFrame
        df = pd.concat(metrics_list, axis=0).reset_index(drop=True)
        df.to_csv(self.reduced_path, index=False)

        return df

    def convert_ratings(self, df_ratings: pd.DataFrame) -> pd.DataFrame:
        """Convert ratings in DataFrame 0 to 5 to 'negative', 'neutral', 'positive'."""

        def conv(num: np.number) -> str:
            num = int(num)

            if num == 3:
                return "neutral"
            if num > 3:
                return "positive"
            return "negative"

        for col in df_ratings.columns:
            if "rating" not in col:
                continue

            df_ratings[col] = df_ratings[col].map(conv)

        return df_ratings

    def gen_metrics(
        self,
        df_senti: pd.DataFrame,
        file_path: Path,
    ) -> pd.DataFrame:
        """Generate F1 score, precision, recall and confusion matrix.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing sentiment ratings generated by local LLM
                and Perplexity API.
            file_path (Path):
                Path object to csv file containing sentiment ratings for
                specific local LLM.

        Returns:
            df_metrics (pd.DataFrame):
                DataFrame with 'model', 'f1_score', 'precision', 'recall',
                'total_time', 'mean_time' and 'cmatrix'.
        """

        # Get model name from model file path
        model_name = file_path.stem.split("_", maxsplit=1)[1]

        # Get true and predicted ratings
        y_true = df_senti["rating"]
        y_pred = df_senti[f"rating_{model_name}"]

        # Convert confusion matrix to a json string
        cmatrix = confusion_matrix(y_true, y_pred, labels=self.ratings)
        cmatrix_str = json.dumps(cmatrix.tolist())

        metrics = {
            "family": file_path.parent.name,
            "model": model_name,
            "f1_score": f1_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "total_time": df_senti["infer_time"].sum(),
            "mean_time": df_senti["infer_time"].mean(),
            "cmatrix": cmatrix_str,
        }

        # Convert to DataFrame
        df_metrics = pd.DataFrame([metrics])

        # Plot and save confusion matrix as heatmap
        self.plot_cmatrix(cmatrix, file_path)

        return df_metrics

    def plot_cmatrix(
        self,
        cmatrix: np.ndarray,
        file_path: Path,
    ) -> None:
        """Plot confusion matrix as seaborn heatmap.

        Args:
            cmatrix (np.ndarray):
                Confusion matrix as numpy array.
            file_path (Path):
                Path object to csv file containing sentiment ratings for
                specific local LLM.
            ratings (list[str]):
                Rating labels (Default: ["negative", "neutral", "positive"]).

        Returns:
            None.
        """

        heatmap = {
            "annot": True,  # Display values in confusion matrix
            "fmt": "d",  # Display values as integer
            "cmap": "Blues",  # Seaborn color map
            "cbar": True,  # Display color bar
        }
        model_name = file_path.stem.split("_", maxsplit=1)[1]

        # Generate labels for ratings 1 to 5
        idx_labels = [f"Act {rating.title()}" for rating in self.ratings]
        col_labels = [f"Pred {rating.title()}" for rating in self.ratings]

        # Convert confusion matrix to DataFrame
        df_cm = pd.DataFrame(cmatrix, index=idx_labels, columns=col_labels)

        # Plot confusion matrix as seaborn heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        ax = sns.heatmap(df_cm, **heatmap)
        ax.set_xlabel("Predicted Sentiment Rating")
        ax.set_ylabel("Actual Sentiment Rating")
        ax.set_title(f"Confusion Matrix for {model_name}")

        # Save figure as png file
        plt.tight_layout()
        fig.savefig(file_path.with_name(f"cm_{model_name}_reduced.png"))
        plt.close()

    def plot_top_n(
        self,
        df_metrics: pd.DataFrame,
        is_reduced: bool = False,
    ) -> None:
        """Plot top N model by selected metric."""

        df = df_metrics.copy()

        # Sort top N model with highest metric
        df = df.sort_values(by=self.metric, ascending=False).reset_index(drop=True)
        df = df.head(self.top_n)

        # Plot bar charts
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=df["model"], y=df[self.metric], ax=ax)

        ax.set_title(f"Top {self.top_n} Local LLM with highest '{self.metric}'")
        ax.set_xlabel("Local LLM")
        ax.set_ylabel(self.metric)
        ax.tick_params(axis="x", rotation=30)

        # Annotate metrics
        for bar in ax.patches:
            ax.annotate(
                round(bar.get_height(), 6),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
            )

        img_name = (
            f"top_{self.top_n}_{self.metric}_reduced.png"
            if is_reduced
            else f"top_{self.top_n}_{self.metric}.png"
        )

        plt.tight_layout()
        plt.savefig(f"{self.overall_dir}/{img_name}")
        plt.close()
