"""Training step"""
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from loguru import logger
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor


def load_model_params() -> Dict:
    """Load train section from params.yaml

    Returns:
        Dict: params to be passed to the classifier constructor
    """
    text = Path("params.yaml").read_text()
    return yaml.safe_load(text).get("train", {})

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
        original_data.columns.str.contains("_name") # sklearn GradientBoostingRegressor does not handle strings
    ]
    original_data = original_data.drop(columns=columns_to_drop)
    X = original_data.drop(columns=["item_cnt_month"])
    y = original_data["item_cnt_month"]
    return X, y

def get_model() -> GradientBoostingRegressor:
    """Load scikit-learn model for training

    Returns:
        GradientBoostingRegressor: scikit-learn classifier
    """
    params = load_model_params()
    logger.info(f"loaded params: {params}")
    model = GradientBoostingRegressor(**params)
    logger.info(f"Model config: {model}")
    return model

def run_training(train_folder: Path) -> GradientBoostingRegressor:
    """Run training

    Args:
        train_folder (Path): folder with input data

    Returns:
        GradientBoostingRegressor: trained model
    """
    original_data = load_df(train_folder)
    X_train, y_train = load_data_for_model(original_data)
    model = get_model()
    model.fit(X=X_train, y=y_train)
    return model

def save_model(model: GradientBoostingRegressor, model_file: Path):
    """Save model in pickle format

    Args:
        model (GradientBoostingRegressor): trained model
        model_file (Path): file to write model to
    """
    model_pickle = pickle.dumps(model)
    model_file.write_bytes(model_pickle)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_folder", type=Path)
    parser.add_argument("--model_file", type=Path,
                        required=False, default=Path("model.pkl"))
    args = parser.parse_args()
    trained_model = run_training(train_folder=args.train_folder)
    save_model(model=trained_model, model_file=args.model_file)
