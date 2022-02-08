# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple, Union

import lightgbm
import mlflow
import numpy as np
import pandas as pd
import typer
from azureml.core import Model, Run, Workspace
from azureml.exceptions import ModelNotFoundException

from sklearn import metrics

METADATA_JSON = "metadata.json"
RESOURCE_DOES_NOT_EXIST = "RESOURCE_DOES_NOT_EXIST"

logger = logging.getLogger("evaluate.py")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class ModelDirection(Enum):
    HIGHER_BETTER = 1
    LOWER_BETTER = 2


def read_dataframe(input_path: Path) -> pd.DataFrame:
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


def load_model_metadata(model_metadata_folder: str) -> Tuple[str, str]:
    model_metadata_path = Path(model_metadata_folder)
    model_metadata_file = model_metadata_path / METADATA_JSON

    with open(model_metadata_file, 'r') as fo:
        metadata = json.load(fo)

    return metadata['run_id'], metadata['model_name']


def write_indicator_file(
        output_folder: str,
        register_model: bool
) -> None:
    """Write an empty file in the output path to indicated if the model should be registered"""
    output_path = Path(output_folder)
    filename = "REGISTER" if register_model else "SKIP"

    (output_path / filename).touch()


def get_model_metrics(model: lightgbm.Booster, data: lightgbm.Dataset, model_name: str) -> dict:
    """Construct a dictionary of metrics for the model"""

    predictions = model.predict(data.data)
    fpr, tpr, thresholds = metrics.roc_curve(data.label, predictions)

    best_threshold = predict_best_threshold(fpr, tpr, thresholds)
    f1_score = metrics.f1_score(data.label, np.where(
        predictions < best_threshold, 0, 1))

    model_metrics = {"auc": metrics.auc(fpr, tpr),
                     "f1-score": f1_score}

    logger.info(f"{model_name} Metrics {model_metrics}")

    return model_metrics


def predict_best_threshold(fpr, tpr, thresholds):
    tnr = 1 - fpr

    bal_acc = (tpr + tnr) / 2
    best_threshold = thresholds[np.argmax(bal_acc)]
    return best_threshold


def challenger_metric_better(champ_metrics: dict,
                             challenger_metrics: dict,
                             metric_name: str,
                             direction: ModelDirection) -> bool:
    if direction == ModelDirection.HIGHER_BETTER:
        return challenger_metrics[metric_name] > champ_metrics[metric_name]
    else:
        return challenger_metrics[metric_name] < champ_metrics[metric_name]


def compare_models(
        champion_model: lightgbm.Booster,
        challenger_model: lightgbm.Booster,
        valid_df: pd.DataFrame,
        comparison_metric: Literal["any", "all", "f1_score", "auc"] = "any"
) -> bool:
    """
    A function to compare the performance of the Champion and Challenger models
    on the validation dataset comparison metrics
    """
    comparison_metrics_directions = {"f1-score": ModelDirection.HIGHER_BETTER,
                                     "auc": ModelDirection.HIGHER_BETTER,
                                     "accuracy": ModelDirection.HIGHER_BETTER}

    # Prep datasets
    features = valid_df.drop(['target', 'id'], axis=1, errors="ignore")
    labels = np.array(valid_df['target'])
    valid_dataset = lightgbm.Dataset(data=features, label=labels)

    # Calculate Champion and Challenger metrics for each
    champion_metrics = get_model_metrics(champion_model, valid_dataset, "Champion")
    challenger_metrics = get_model_metrics(challenger_model, valid_dataset, "Challenger")

    if comparison_metric not in ['any', 'all']:
        logger.info(f"Champion performance for {comparison_metric}: {champion_metrics[comparison_metric]}")
        logger.info(f"Challenger performance for {comparison_metric}: {challenger_metrics[comparison_metric]}")

        register_model = challenger_metric_better(champ_metrics=champion_metrics,
                                                  challenger_metrics=challenger_metrics,
                                                  metric_name=comparison_metric,
                                                  direction=comparison_metrics_directions[comparison_metric])
    else:
        comparison_results = {metric: challenger_metric_better(champ_metrics=champion_metrics,
                                                               challenger_metrics=challenger_metrics,
                                                               metric_name=metric,
                                                               direction=comparison_metrics_directions[metric])
                              for metric in champion_metrics.keys()}

        if comparison_metric == "any":
            register_model = any(comparison_results.values())
            if register_model:
                positive_results = [metric for metric, result in comparison_results.items() if result]
                for metric in positive_results:
                    logger.info(f"Challenger Model performed better for '{metric}' on validation data")
            else:
                logger.info("Champion model performed better for all metrics on validation data")
        else:
            register_model = all(comparison_results.values())
            if register_model:
                logger.info("Challenger model performed better on all metrics on validation data")
            else:
                negative_ressults = [metric for metric, result in comparison_results.items() if not result]
                for metric in negative_ressults:
                    logger.info(f"Champion Model performed better for '{metric}' on validation data")

    return register_model


def load_champion_model(
        model_name: str,
        register_model: bool
) -> Tuple[Union[lightgbm.Booster, None], bool]:
    """
    Load the champion model as the currently registered model of 'model_name' and the highest version number
    """
    run = Run.get_context()
    ws: Workspace = run.experiment.workspace

    # Load Champion Model
    # If the champion model doesn't exist, recommend register model
    try:
        champ_temp_dir = tempfile.mkdtemp()
        champion_model = Model(ws, model_name)
        champion_model.download(target_dir=champ_temp_dir)
        champion_model = mlflow.lightgbm.load_model(os.path.join(champ_temp_dir, "model"))

    except ModelNotFoundException:
        logger.info(f"No model named '{model_name}' currently in registry - recommending model registration")
        register_model = True
        champion_model = None

    return champion_model, register_model


def load_challenger_model(
        model_name: str,
        run_id: str
) -> lightgbm.Booster:
    """Load challenger model from this Pipeline"""
    challenger_model_uri = f"runs:/{run_id}/model"
    challenger_model = mlflow.lightgbm.load_model(challenger_model_uri)
    return challenger_model


def main(
    model_metadata: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False),
    recommendation_folder: Path = typer.Option(..., exists=False, dir_okay=True, file_okay=False),
    validation_data_path: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=True),
):
    """
    Download the currently registered model from AML Model Registry and compare the model performance
    on a standard dataset.

    The "Champion" model - is the model currently registered and in production. The "Challenger" model is the model
    currently

    If the challenger model wins, then promote the model to production.
    Otherwise, keep the champion model in production.
    """
    # TODO: Implement OpenCensus and Shell logging

    recommendation_folder.mkdir(parents=True, exist_ok=True)
    register_model = False

    # Load the RunID and Model name from the model training step
    run_id, model_name = load_model_metadata(model_metadata)

    # Load champion model from the Model Registry
    champion_model, register_model = load_champion_model(model_name, register_model)

    # Load the challenger model from 'Train Model' step
    challenger_model = load_challenger_model(model_name, run_id)

    valid_df = read_dataframe(validation_data_path)

    # If the champion model exists, then run the compare model function
    if champion_model is not None:
        register_model = compare_models(champion_model=champion_model,
                                        challenger_model=challenger_model,
                                        valid_df=valid_df,
                                        comparison_metric="all")

    logger.info(f"Is Model Registration Recommended?: {register_model}")
    # Write the indicator file to pass along to the "Register Model" step.
    # The folder will either contain an empty file that says "REGISTER" or an empty file that says "SKIP"
    write_indicator_file(output_folder=recommendation_folder, register_model=register_model)


if __name__ == "__main__":
    typer.run(main)
