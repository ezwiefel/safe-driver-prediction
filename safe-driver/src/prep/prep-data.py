# Copyright (c) 2022 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import List
from pathlib import Path

import pandas as pd
import typer
from pandas import DataFrame


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


def main(
    drop_columns: List[str],
    input_data: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=True),
    labels_output: Path = typer.Option(..., dir_okay=True, file_okay=False),
    features_output: Path = typer.Option(..., dir_okay=True, file_okay=False),
    label_column: str = 'target',
):
    data_df = read_dataframe(input_data)

    labels_output.mkdir(exist_ok=True, parents=True)
    features_output.mkdir(exist_ok=True, parents=True)

    if not isinstance(drop_columns, list):
        drop_columns = list(drop_columns)
    drop_columns.append(label_column)

    labels = data_df[[label_column]]
    features = data_df.drop(drop_columns, axis=1)

    labels.to_parquet(labels_output / 'labels.parquet')
    features.to_parquet(features_output / 'features.parquet')


if __name__ == '__main__':
    typer.run(main)
