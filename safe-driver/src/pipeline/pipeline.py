# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
from typing import Tuple
from pathlib import Path

import typer
from azureml.core import (ComputeTarget, Dataset,
                          RunConfiguration, Workspace)
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.experiment import Experiment
from azureml.pipeline.core import (Pipeline, PipelineData, PipelineParameter,
                                   PublishedPipeline)
from azureml.pipeline.steps import PythonScriptStep

CLI_AUTH = AzureCliAuthentication()
# noinspection PyTypeChecker
WS = Workspace.from_config(auth=CLI_AUTH)


# noinspection PyTypeChecker
def create_data_prep_step(
    prep_directory: Path,
    input_dataset: Dataset,
    compute: ComputeTarget,
    debug_run: bool,
    run_config: RunConfiguration
) -> Tuple[PythonScriptStep, PipelineData, PipelineData]:
    feature_data = PipelineData(
        name="features",
        datastore=WS.get_default_datastore(),
        is_directory=True
    )
    label_data = PipelineData(
        name="labels",
        datastore=WS.get_default_datastore(),
        is_directory=True
    )
    input = input_dataset.as_named_input("raw_data")
    prep_data_step = PythonScriptStep(
        name="Prep Data",
        script_name="prep-data.py",
        source_directory=prep_directory,
        compute_target=compute,
        outputs=[feature_data, label_data],
        allow_reuse=debug_run,
        arguments=["--labels-output", label_data,
                   "--features-output", feature_data,
                   "--input-data", input.as_mount(),
                   "--label-column", "target",
                   "id"],
        runconfig=run_config
    )

    return prep_data_step, feature_data, label_data


# noinspection PyTypeChecker
def create_train_model_step(
    train_script_path: Path,
    feature_data: PipelineData,
    label_data: PipelineData,
    compute: ComputeTarget,
    debug_run: bool,
    run_config: RunConfiguration
) -> Tuple[PythonScriptStep, PipelineData]:

    metadata_folder = PipelineData(
        name="metadata",
        datastore=WS.get_default_datastore(),
        is_directory=True
    )

    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        source_directory=train_script_path,
        compute_target=compute,
        inputs=[feature_data, label_data],
        outputs=[metadata_folder],
        allow_reuse=debug_run,
        arguments=[
            "--output-folder", metadata_folder,
            "--feature-data", feature_data.as_mount(),
            "--label-data", label_data.as_mount(),
            "--random-state", "11235"
        ],
        runconfig=run_config
    )

    return train_step, metadata_folder


# noinspection PyTypeChecker
def create_evaluate_model_step(
    evaluate_script_path: Path,
    model_metadata_folder: PipelineData,
    compute: ComputeTarget,
    validation_data: Dataset,
    debug_run: bool,
    run_config: RunConfiguration
) -> Tuple[PythonScriptStep, PipelineData]:
    """
    Creates "Evaluate Model" Step
    """
    output_folder = "./outputs"
    output_data = PipelineData(
        name="recommendation",
        datastore=WS.get_default_datastore(),
        is_directory=True,
        output_path_on_compute=output_folder,
        output_mode="upload"
    )

    eval_step = PythonScriptStep(
        name="Evaluate Model",
        script_name="evaluate.py",
        source_directory=evaluate_script_path,
        compute_target=compute,
        inputs=[model_metadata_folder],
        outputs=[output_data],
        arguments=[
            "--model-metadata", model_metadata_folder.as_mount(),
            "--recommendation-folder", output_folder,
            "--validation-data-path", validation_data.as_named_input("validation_data").as_mount()
        ],
        allow_reuse=debug_run,
        runconfig=run_config
    )

    return eval_step, output_data


# noinspection PyTypeChecker
def create_register_model_step(
    register_script_path: Path,
    model_folder: PipelineData,
    register_model_folder: PipelineData,
    compute: ComputeTarget,
    debug_run: bool,
    run_config: RunConfiguration
) -> PythonScriptStep:
    """
    Creates "Register Model" PythonScriptStep
    """
    force_param = PipelineParameter(name="force_registration", default_value="False")
    skip_param = PipelineParameter(name="skip_registration", default_value="False")

    register_step = PythonScriptStep(
        name="Register Model",
        script_name="register.py",
        source_directory=register_script_path,
        compute_target=compute,
        inputs=[
            model_folder,
            register_model_folder
        ],
        arguments=[
            "--force", force_param,
            "--skip", skip_param,
            "--model-metadata", model_folder.as_mount(),
            "--recommendation-folder", register_model_folder.as_mount()
        ],
        allow_reuse=debug_run,
        runconfig=run_config
    )

    return register_step


def create_pipeline(
    prep_script_path: Path,
    train_script_path: Path,
    evaluate_script_path: Path,
    register_script_path: Path,
    debug_run: bool,
    aml_compute: str,
    input_dataset: str,
    validation_dataset: str,
    run_config: RunConfiguration
) -> Pipeline:
    """
    Creates the overall Pipeline

    Dataset -> Convert to Parquet -> Create Features -> (Train Model -> Evaluate Model) -> Register Model
    """
    cpu_cluster = WS.compute_targets[aml_compute]

    raw_csv_data = WS.datasets[input_dataset]
    validation_data = WS.datasets[validation_dataset]

    # Prep Data step
    prep_step, features, labels = create_data_prep_step(
        prep_directory=prep_script_path,
        compute=cpu_cluster,
        input_dataset=raw_csv_data,
        debug_run=debug_run,
        run_config=run_config
    )

    # Train Model Step
    train_step, metadata_folder = create_train_model_step(
        train_script_path=train_script_path,
        feature_data=features,
        label_data=labels,
        compute=cpu_cluster,
        debug_run=debug_run,
        run_config=run_config
    )

    # Evaluate Model Step
    eval_step, recommendation_folder = create_evaluate_model_step(
        evaluate_script_path=evaluate_script_path,
        model_metadata_folder=metadata_folder,
        compute=cpu_cluster,
        validation_data=validation_data,
        debug_run=debug_run,
        run_config=run_config
    )

    # Register Model Step
    reg_step = create_register_model_step(
        register_script_path=register_script_path,
        model_folder=metadata_folder,
        register_model_folder=recommendation_folder,
        compute=cpu_cluster,
        debug_run=debug_run,
        run_config=run_config
    )

    return Pipeline(WS, steps=[reg_step])


def main(
    prep_script_path: Path = typer.Option(Path("./safe-driver/src/prep"), exists=True, dir_okay=True, file_okay=False),
    train_script_path: Path = typer.Option(Path("./safe-driver/src/train"),
                                           exists=True, dir_okay=True, file_okay=False),
    evaluate_script_path: Path = typer.Option(Path("./safe-driver/src/evaluate"),
                                              exists=True, dir_okay=True, file_okay=False),
    register_script_path: Path = typer.Option(Path("./safe-driver/src/register"),
                                              exists=True, dir_okay=True, file_okay=False),
    force_model_register: bool = typer.Option(False, "--force-registration",
                                              help="Force the model registration. Ignores model "
                                                   "performance against the existing model"),
    skip_model_register: bool = typer.Option(False, '--skip-registration',
                                             help="Skip the model registration. Ignores model "
                                                  "performance against the existing model"),
    submit_pipeline: bool = typer.Option(True, "--skip-submit-pipeline",
                                         help="Submit the pipeline for a run"),
    publish_pipeline: bool = typer.Option(False, '--publish-pipeline',
                                          help="Publish the pipeline"),
    experiment_name: str = typer.Option("pipeline-test",
                                        help="If submitting pipeline, submit under this experiment name"),
    debug_run: bool = typer.Option(False, "--debug",
                                   help="Reuse pipeline steps as a part of debugging"),
    environment: str = "lightgbm",
    aml_compute_name: str = 'cpu-cluster',
    input_dataset_name: str = 'safe_driver',
    validation_dataset_name: str = 'safe_driver_validation',
    run_id: str = None,
    run_attempt: int = None,
) -> None:
    """Submit and/or publish the pipeline"""

    lgbm_environment = WS.environments[environment]
    run_config = RunConfiguration()
    run_config.environment = lgbm_environment

    pipeline: Pipeline = create_pipeline(
        prep_script_path=prep_script_path,
        train_script_path=train_script_path,
        evaluate_script_path=evaluate_script_path,
        register_script_path=register_script_path,
        debug_run=debug_run,
        aml_compute=aml_compute_name,
        input_dataset=input_dataset_name,
        validation_dataset=validation_dataset_name,
        run_config=run_config,
    )
    pipeline.validate()

    if run_id or run_attempt:
        tags = {"run": run_id, "attempt": run_attempt}
    else:
        tags = None

    if submit_pipeline and not publish_pipeline:
        exp = Experiment(WS, experiment_name)
        exp.submit(pipeline,
                   pipeline_parameters={"force_registration": str(force_model_register),
                                        "skip_registration": str(skip_model_register)},
                   tags=tags)

    if publish_pipeline:
        published_pipeline: PublishedPipeline = pipeline.publish(
            name="Driver Safety Pipeline",
            description="Training Pipeline for new driver safety model"
        )

        if submit_pipeline:
            published_pipeline.submit(
                workspace=WS,
                experiment_name=experiment_name,
                pipeline_parameters={"force_registration": str(force_model_register),
                                     "skip_registration": str(skip_model_register)}
            )

        sys.stdout.write(published_pipeline.id)


if __name__ == "__main__":
    typer.run(main)
