# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging
from distutils.util import strtobool
from pathlib import Path
from typing import Tuple

import typer
import mlflow

METADATA_JSON = "metadata.json"

logger = logging.getLogger("register_model.py")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_indicator_file(recommendation_folder: str) -> bool:
    """Read an empty file left in the output path that indicated if the model should be registered"""
    output_path = Path(recommendation_folder)

    return len(list(output_path.glob("REGISTER"))) > 0


def load_model_metadata(model_metadata_folder: str) -> Tuple[str, str]:
    model_metadata_path = Path(model_metadata_folder)
    model_metadata_file = model_metadata_path / METADATA_JSON

    with open(model_metadata_file, 'r') as fo:
        metadata = json.load(fo)

    return metadata['run_id'], metadata['model_name']


def main(
    force: str = "False",
    skip: str = "False",
    model_metadata: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False),
    recommendation_folder: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False),
):

    # print(f"force: {force}")
    # print(f"skip: {skip}")

    force = bool(strtobool(force))
    skip = bool(strtobool(skip))

    # print(f"force: {force}")
    # print(f"skip: {skip}")

    if force and skip:
        raise ValueError("Model registration cannot be both forced and skipped")

    # TODO: Implement OpenCensus and Shell logging
    logger.info("Logging started")
    print("Logging started")

    run_id, model_name = load_model_metadata(model_metadata)
    logger.info(f"Model Run ID: {run_id}")
    logger.info(f"Model Name: {model_name}")

    register_recommended = read_indicator_file(recommendation_folder=recommendation_folder)

    logger.info(f"Is Registration Recommended?: {register_recommended}")
    logger.info(f"Is Registration Forced?: {force}")
    logger.info(f"Is Registration Skipped?: {skip}")

    # If force, then register
    # If skip, then don't register
    # Otherwise, look for the indicator file in the 'register-model-folder' location
    register_recommended = True if force else register_recommended
    register_recommended = False if skip else register_recommended

    if register_recommended:
        # run = Run.get_context()
        # challenger_run = Run(experiment=run.experiment, run_id=run_id)

        # challenger_run.register_model(model_name=model_name, model_path="model", )

        challenger_model_uri = f"runs:/{run_id}/model"
        logger.info(f"Model Registered(URI={challenger_model_uri}, Name={model_name})")
        mlflow.register_model(model_uri=challenger_model_uri, name=model_name)


if __name__ == "__main__":
    typer.run(main)
