"""scripts/00_download_data.py — symlink/copy dataset into data/raw/"""

import shutil
import sys
from pathlib import Path

SRC = Path("kaggle_dataset/twcs/twcs.csv")
DST = Path("data/raw/twcs.csv")


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        print("Download twcs.csv from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        sys.exit(1)

    DST.parent.mkdir(parents=True, exist_ok=True)

    if DST.exists() or DST.is_symlink():
        print(f"Already exists: {DST}")
        return

    try:
        DST.symlink_to(SRC.resolve())
        print(f"Symlinked: {DST} -> {SRC.resolve()}")
    except OSError:
        shutil.copy2(SRC, DST)
        print(f"Copied: {SRC} -> {DST}")

    import csv
    with open(DST, newline="") as f:
        reader = csv.DictReader(f)
        rows = sum(1 for _ in reader)
    print(f"Rows in dataset: {rows:,}")


if __name__ == "__main__":
    main()
