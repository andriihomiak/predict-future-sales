import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def load_model(model_file: Path) -> GradientBoostingRegressor:
    model: GradientBoostingRegressor = pickle.loads(model_file.read_bytes())
    return model


def load_df(folder: Path) -> DataFrame:
    """Load prepared data into dataframe from folder

    Args:
        folder (Path): folder containing data.csv

    Returns:
        DataFrame: prepared dataframe
    """
    return pd.read_csv(folder/"data.csv")


def load_data_for_model(original_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Load model-specific version of data (with addition type transformations etc.)

    Args:
        original_data (DataFrame): original dataframe

    Returns:
        Tuple[DataFrame, DataFrame]: (X, y) dataframes ready for model.fit()
    """
    columns_to_drop = original_data.columns[
        # sklearn GradientBoostingRegressor does not handle strings
        original_data.columns.str.contains("_name")
    ]
    original_data = original_data.drop(columns=columns_to_drop)
    X = original_data.drop(columns=["item_cnt_month"])
    y = original_data["item_cnt_month"]
    return X, y


def load_test_range(test_folder: Path):
    return pd.read_csv(test_folder/"test.csv", index_col=["shop_id", "item_id"])


def extend_target_df(test_range: DataFrame, val_df: DataFrame, prediction: np.ndarray) -> DataFrame:
    target_df = test_range.join(val_df.set_index(["shop_id", "item_id"]).assign(prediction=prediction)).assign(
        date_block_num=(24 + 9), 
        item_cnt_month=lambda df: df.item_cnt_month.fillna(0),
        prediction=lambda df: df.prediction.fillna(0),
        date_year=2015, 
        date_month=9,
    )
    return target_df


def evaluate_model(model_file: Path, val_folder: Path, test_folder: Path) -> Dict:
    """Evaluate model

    Args:
        model_file (Path): path to mode pickle file

    Returns:
        Dict: metrics
    """
    model = load_model(model_file)
    val_df = load_df(val_folder)
    X_val, _ = load_data_for_model(val_df)
    prediction = model.predict(X_val)
    test_range = load_test_range(test_folder)
    extended = extend_target_df(test_range, val_df, prediction)
    return {
        "rmse": rmse(extended.item_cnt_month, extended.prediction)
    }


def write_metrics(metrics: Dict, file: Path):
    file.write_text(json.dumps(metrics))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("val_folder", type=Path)
    parser.add_argument("test_folder", type=Path)
    parser.add_argument("--model_file", type=Path, default=Path("model.pkl"))
    parser.add_argument("--metrics_file", type=Path,
                        default=Path("metrics.json"))

    args = parser.parse_args()

    metrics = evaluate_model(model_file=args.model_file, 
        val_folder=args.val_folder,
        test_folder=args.test_folder)
    write_metrics(metrics=metrics, file=args.metrics_file)