# Copyright (c) 2022 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pathlib import Path
import mlflow
from typing import Union
import typer


def write_indicator_file(output_folder: str, register_model: bool):
    """Write an empty file in the output path to indicated if the model should be registered"""
    output_path = Path(output_folder)
    filename = "REGISTER" if register_model else "SKIP"

    (output_path / filename).touch()


def compare_model_performance(past_tags: Union[dict, None], current_metrics: dict,
                              interested_metrics=['auc', 'f1-score']) -> bool:
    # If there was no past model, the metrics will be None

    print("************************")
    print("*** MODEL EVALUATION ***")
    print("************************")

    if past_tags is None:
        print("No previous model", "Skipping evaluation", sep="\n")
        return True

    for metric in interested_metrics:
        new_metric = float(current_metrics.get(metric, 0))
        prev_metric = float(past_tags.get(metric))

        print(f"{metric}: New: {new_metric} - Previous {prev_metric}")

        if new_metric < prev_metric:
            print()
            print("  REGISTRATION ABORTED")
            print(f"  {metric.upper()} WORSE")
            print(f"************************")
            return False

        print("  REGISTRATION RECOMMENDED")
        print(f"************************")
    return True


def assess_skip_evaluation(force: bool, skip: bool) -> Union[None, bool]:
    """If force or skip are passed, write the appropriate file and exit"""
    if skip:
        print("MODEL REGISTRATION SKIPPED")
        print("SKIPPING EVALUATION")
        return False
    elif force:
        print("MODEL REGISTRATION FORCED")
        print("SKIPPING EVALUATION")
        return True
    else:
        return None


def main(
    metadata: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=True),
    force: bool = typer.Option(False, "--force"),
    skip: bool = typer.Option(False, "--skip"), 
    model_name: str = "SafeDriverModel",
    output_folder: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False)):
    """
    Using the parent run as a "storage area" for model metrics, 
    we compare the past model performance to the current model performance.

    Here, we aren't recomputing the model metrics 
    (although, this might be something to consider for the future - compute model
    metrics for both models on a specific dataset)
    """
    register_model = assess_skip_evaluation(force=force, skip=skip)

    if register_model is None:
        champion_model =         
        
        # Fetch the tags from the last model that show model metrics
        past_model_tags = get_historic_model_tags(model_name, ws)

        # Fetch the metrics from the current model
        current_metrics = run.parent.get_metrics()
        register_model = compare_model_performance(past_model_tags, current_metrics)

    write_indicator_file(output_folder=output_folder, register_model=register_model)


if __name__ == "__main__":
    main()
