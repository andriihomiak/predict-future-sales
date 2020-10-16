from pathlib import Path

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("val_dir", type=Path)
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("--out_dir", type=Path, required=False, default=Path("data/prepared"))
    args = parser.parse_args()

    (args.out_dir/"train").mkdir(exist_ok=True, parents=True)
    (args.out_dir/"val").mkdir(exist_ok=True, parents=True)
    (args.out_dir/"test").mkdir(exist_ok=True, parents=True)
