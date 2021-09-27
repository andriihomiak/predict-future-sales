from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from loguru import logger


def convert_csvs_to_hdf(source: Path, destination: Path):
    # Convert files that require no processing and optionally can be renamed
    name_mapping = {
        "item_categories": "item_categories",
        "items": "items",
        "shops": "shops",
        "sales_train": "train",
        "test": "submission_target",
    }
    destination.mkdir(exist_ok=True, parents=True)
    target_df: pd.DataFrame = None
    sales_df: pd.DataFrame = None
    for file in name_mapping:
        source_file = source/f"{file}.csv"
        target_file = destination/f"{name_mapping[file]}.hdf"
        logger.info(f"Converting {source_file} -> {target_file}")
        df = pd.read_csv(source_file)
        df.to_hdf(target_file, key=file)
        if file == "test":
            target_df = df
        if file == "sales_train":
            sales_df = df
    
    # Create test file with only item-shop pairs that have occured previously
    all_pairs = set((row.shop_id, row.item_id) for row in sales_df[["shop_id", "item_id"]].itertuples())
    test_df = target_df.apply(lambda row: (row.shop_id, row.item_id) in all_pairs, axis=1)
    test_df.to_hdf(destination/"test.hdf", key="test")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", type=Path)
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    convert_csvs_to_hdf(source=args.src, destination=args.dest)

