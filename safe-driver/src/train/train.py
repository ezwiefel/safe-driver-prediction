# Copyright (c) 2022 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
from pathlib import Path
from typing import Dict

import typer
import lightgbm
import mlflow
import numpy as np
import pandas as pd
from lightgbm import Dataset, Booster
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split

METADATA_JSON = "metadata.json"
MODEL_NAME = "SafeDriverModel"


def read_dataframe(input_path: Path) -> DataFrame:
    """
    Read a file (CSV or Parquet) and return a dataframe.
    If a directory is passed, look for parquet, if not available, look for CSV
    (Based on file extension)
    """
    if input_path.is_dir():
        parquet_files = list(input_path.glob("*.parquet"))
        csv_files = list(input_path.glob("*.csv"))
        if len(parquet_files) > 0:
            input_path = parquet_files[0]
        elif len(csv_files) > 0:
            input_path = csv_files[0]
        else:
            raise FileNotFoundError("No CSV or Parquet files found")

    file_suffix = input_path.suffix
    if file_suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif file_suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise FileNotFoundError("File is not CSV or Parquet")
    return df


def split_data(label_df: DataFrame, feature_df: DataFrame, test_size: float, random_state: int) -> Dict[str, Dataset]:
    """Split feature/label dataframes into training and validation datasets"""

    features_train, features_valid, labels_train, labels_valid = train_test_split(  # noqa: E501
        feature_df, label_df, test_size=test_size, random_state=random_state)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(
        features_valid, label=labels_valid, free_raw_data=False)

    return {"train": train_data, "valid": valid_data}


def train_model(data: Dict[str, Dataset], parameters: dict) -> Booster:
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    model = lightgbm.train(params=parameters,
                           train_set=data["train"],
                           valid_sets=data["valid"],
                           num_boost_round=500,
                           early_stopping_rounds=20)

    return model


def predict_best_threshold(fpr, tpr, thresholds):
    tnr = 1 - fpr

    bal_acc = (tpr + tnr) / 2
    best_threshold = thresholds[np.argmax(bal_acc)]
    return best_threshold


def get_model_metrics(model: Booster, data: Dataset) -> dict:
    """Construct a dictionary of metrics for the model"""

    predictions = model.predict(data.data)
    fpr, tpr, thresholds = metrics.roc_curve(data.label, predictions)

    best_threshold = predict_best_threshold(fpr, tpr, thresholds)
    f1_score = metrics.f1_score(data.label, np.where(
        predictions < best_threshold, 0, 1))

    model_metrics = {"auc": metrics.auc(fpr, tpr),
                     "f1-score": f1_score}
    print(model_metrics)
    return model_metrics


def save_model_metadata(run_id: str, model_name: str, output_file: Path) -> None:
    with open(output_file, 'w+') as file_handler:
        metadata = dict(
            run_id=run_id,
            model_name=model_name
        )
        json.dump(metadata, file_handler)


def main(
        label_data: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=True),
        feature_data: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=True),
        output_folder: Path = typer.Option(..., file_okay=False, dir_okay=True),
        learning_rate: float = 0.03,
        boosting_type: str = "gbdt",
        objective: str = "binary",
        metric: str = "auc",
        sub_feature: float = 0.7,
        num_leaves: int = 60,
        min_data: int = 100,
        min_hessian: int = 1,
        verbose: int = 0,
        random_state: int = 0,
        test_size: float = 0.2,
) -> None:

    # Prepare parameters dict
    model_params = dict(
        learning_rate=learning_rate,
        boosting_type=boosting_type,
        objective=objective,
        metric=metric,
        sub_feature=sub_feature,
        num_leaves=num_leaves,
        min_data=min_data,
        min_hessian=min_hessian,
        verbose=verbose
    )
    # TODO: Implement OpenCensus and Shell logging

    # load the safe driver prediction dataset and
    # split data into training and valdation datasets
    label_df = read_dataframe(label_data)
    feature_df = read_dataframe(feature_data)
    datasets = split_data(feature_df=feature_df,
                          label_df=label_df,
                          test_size=test_size, 
                          random_state=random_state)

    with mlflow.start_run() as run:

        # Autolog the parameters to Azure Machine Learning
        mlflow.lightgbm.autolog()

        # Train the model and calculate model metrics
        _ = train_model(datasets, model_params)

        # Save the model metadata to a
        output_path = Path(output_folder)
        metadata_path = output_path / METADATA_JSON

        # Save the Run ID into the output folder along with the model name
        save_model_metadata(
            run_id=run.info.run_id,
            model_name=MODEL_NAME,
            output_file=metadata_path
        )


if __name__ == "__main__":
    typer.run(main)
