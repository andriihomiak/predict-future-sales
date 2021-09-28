import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from hyperopt import hp, fmin, tpe
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5


def load_data(data_folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # TODO: move to feature generation stage
    train = pd.read_hdf(data_folder/"train/data.hdf")
    val = pd.read_hdf(data_folder/"val/data.hdf")
    X_train = train.drop(
        columns=["item_cnt_month", "item_category_name", "item_name"])
    X_val = val.drop(
        columns=["item_cnt_month", "item_category_name", "item_name"])
    y_train = train.item_cnt_month
    y_val = val.item_cnt_month
    return X_train, X_val, y_train, y_val


def tune_hyperparameters(data: Path, metrics_file: Path, output_file: Path):
    X_train, X_val, y_train, y_val = load_data(data_folder=data)

    def evaluate_model(params: Dict) -> float:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        rmse = root_mean_squared_error(y_val, model.predict(X_val))
        return rmse

    def minimization_objective(params: Dict) -> float:
        return -evaluate_model(params)

    space = {
        "random_state": 42,
        "n_estimators": 20,
        "n_jobs": -1,
    }

    # TODO: use hyperopt
    # best_params = fmin(minimisation_objective,
    #                       space=space,
    #                       algo=tpe.suggest)

    best_params = space

    model_metrics = {
        "rmse": evaluate_model(best_params),
    }

    output_file.write_text(json.dumps(best_params))
    metrics_file.write_text(json.dumps(model_metrics))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tune_hyperparameters(data=args.data,
                         metrics_file=args.metrics,
                         output_file=args.output)
