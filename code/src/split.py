import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from argparse import ArgumentParser
from yaml import safe_load
from pathlib import Path


def load_params(path: Optional[Path] = Path("params.yaml")) -> Dict:
    params = safe_load(path.read_text())["validation_split"]
    return params


def split_data(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_start = params["train_start_date"]
    train_end = params["train_end_date"]
    val_end = params["val_end_date"]
    train_range = pd.date_range(train_start, train_end, closed="left")
    val_range = pd.date_range(train_end, val_end, closed="left")
    data = df.assign(date=lambda df: pd.to_datetime(df.date))
    train_data = data.query("date in @train_range")
    val_data = data.query("date in @val_range")
    return train_data, val_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file",
                        help="File to read historical data from")
    parser.add_argument("--params",
                        type=Path,
                        required=False,
                        default=Path("params.yaml"),
                        help="path to params.yaml")
    parser.add_argument("--out_dir",
                        type=Path,
                        required=False,
                        help="path to output folder")
    args = parser.parse_args()
    params = load_params(args.params)
    train, val = split_data(pd.read_hdf(args.file), params)
    (args.out_dir/"train").mkdir(exist_ok=True, parents=True)
    (args.out_dir/"val").mkdir(exist_ok=True, parents=True)
    train.to_hdf(args.out_dir/"train"/"sales.hdf", key="sales")
    val.to_hdf(args.out_dir/"val"/"sales.hdf", key="sales")
