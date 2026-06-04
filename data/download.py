"""Download the public ETTh1 dataset (no auth required).

    python data/download.py

Writes ``data/raw/ETTh1.csv``. If the download fails (offline), the pipeline
falls back to the bundled sample so tests still run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
DEST = Path(__file__).resolve().parent / "raw" / "ETTh1.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(URL, timeout=30) as resp:
            DEST.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed: {exc}\nURL: {URL}")
    rows = sum(1 for _ in DEST.open()) - 1
    print(f"Saved {DEST} ({rows} rows)")


if __name__ == "__main__":
    main()
