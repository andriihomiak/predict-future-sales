from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--model_file", type=Path, required=False, default=Path("model.pkl"))
    args = parser.parse_args()
    model_file: Path = args.model_file
    model_file.write_text("")