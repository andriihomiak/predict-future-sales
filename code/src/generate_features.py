"""Feature generation from sales and item info"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame


def get_item_info_df(src_folder: Path) -> DataFrame:
    """Get DataFrame containing information about items (category, name etc.)

    Args:
        src_folder (Path): folder containing items.csv and item_categories.csv

    Returns:
        DataFrame: resulting dataframe
    """
    items_df = pd.read_csv(src_folder/"items.csv")
    item_categories_df = pd.read_csv(src_folder/"item_categories.csv")
    items_info_df = items_df.merge(item_categories_df, on="item_category_id")
    return items_info_df


def load_sales_from_folder(folder: Path) -> DataFrame:
    """Load sales dataframe from given folder

    Args:
        folder (Path): folder with sales.csv

    Returns:
        DataFrame: sales dataframe
    """
    return (pd.read_csv(folder/"sales.csv")
            .assign(date=lambda df: pd.to_datetime(df.date)))


def aggregate_months(sales_df: DataFrame, item_info_df: DataFrame) -> DataFrame:
    """Aggregate month records for every month in df and add item info

    Args:
        sales_df (DataFrame): dataframe with sales info
        item_info_df (DataFrame): dataframe with items info

    Returns:
        DataFrame: aggregated dataframe
    """
    return (sales_df.query("item_cnt_day > 0")
            .assign(
            date_month=lambda df: df.date.dt.month,
            date_year=lambda df: df.date.dt.year,
            )
            .assign(
            date_block_num=lambda df: (
                df.date_year - 2013) * 12 + df.date_month,
            )
            .assign(date_block_num=lambda df: df.date_block_num.astype(int))
            .groupby(["shop_id", "item_id", "date_block_num"])
            .agg({
                "item_cnt_day": sum,
                "date_year": lambda r: r.iloc[0],
                "date_month": lambda r: r.iloc[0],
                "item_price": np.mean
            })
            .rename(columns={"item_cnt_day": "item_cnt_month"})
            .reset_index()
            .sort_values(["date_block_num", "shop_id", "item_id"])
            .assign(item_cnt_month=lambda df: df.item_cnt_month.clip(0, 20))
            .merge(item_info_df, on="item_id")
            )


def prepare_train_val(train_folder: Path, val_folder: Path,
                      src_folder: Path) -> Tuple[DataFrame, DataFrame]:
    """Prepare train and val dataframes

    Args:
        train_folder (Path): folder containing train sales.csv
        val_folder (Path): folder containing val sales.csv
        src_folder (Path): folder with items.csv and item_categories.csv

    Returns:
        Tuple[DataFrame, DataFrame]: (train_df, val_df)
    """
    item_info = get_item_info_df(src_folder)
    train_sales = load_sales_from_folder(train_folder)
    val_sales = load_sales_from_folder(val_folder)

    train_df = aggregate_months(train_sales, item_info)
    val_df = aggregate_months(val_sales, item_info)
    return train_df, val_df


def write_to_output_folder(train_df: DataFrame, val_df: DataFrame,
                           test_df: DataFrame, out_folder: Path):
    """Write datagrames to output folder

    Args:
        train_df (DataFrame): train dataframe
        val_df (DataFrame): val dataframe
        test_df (DataFrame): test dataframe
        out_folder (Path): folder to write to
    """
    def ensure_exists(kind: str) -> Path:
        folder = out_folder/kind
        folder.mkdir(exist_ok=True, parents=True)
        return folder

    train_folder = ensure_exists("train")
    val_folder = ensure_exists("val")
    test_folder = ensure_exists("test")

    train_df.to_csv(train_folder/"data.csv", index=False)
    val_df.to_csv(val_folder/"data.csv", index=False)
    # test_df.to_csv(test_folder/"data.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_folder", type=Path)
    parser.add_argument("val_folder", type=Path)
    parser.add_argument("src_folder", type=Path)
    parser.add_argument("--out_folder", type=Path,
                        required=False, default=Path("data/prepared"))
    args = parser.parse_args()

    train, val = prepare_train_val(train_folder=args.train_folder,
                                         val_folder=args.val_folder,
                                         src_folder=args.src_folder)

    write_to_output_folder(train_df=train,
                           val_df=val,
                           test_df=None,
                           out_folder=args.out_folder)
