#!/usr/bin/env python3
"""Build TestData/ from integer_dataset using column specs in integer_datasets.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ICDE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ICDE))

from integer_datasets import (
    ALL_INTEGER_DATASET_COLUMNS,
    ALL_INTEGER_DATASET_MAPPING,
    DEFAULT_INTEGER_MAX_VALUES,
    HEADERLESS_INTEGER_FILES,
    INTEGER_DATASET_COLUMNS,
    INTEGER_DATASET_MAPPING,
    INTEGER_SRC,
    LARGE_INTEGER_MAX_VALUES,
    PAPER_TABLE_DATASET_FILES,
)

TEST_DATA = ICDE / 'TestData'


def max_values_for(name: str, cli_default: int) -> int:
    if name in LARGE_INTEGER_MAX_VALUES:
        return LARGE_INTEGER_MAX_VALUES[name]
    return cli_default


def extract_values(path: Path, columns: list[int] | None, max_values: int) -> list[int]:
    if path.name in HEADERLESS_INTEGER_FILES:
        values: list[int] = []
        with path.open(encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    values.append(int(float(s)))
                except ValueError:
                    continue
                if len(values) >= max_values:
                    break
        return values

    nrows = max_values + 1 if max_values else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    if columns is None:
        cols = list(df.columns)
    else:
        cols = [df.columns[i] for i in columns if i < len(df.columns)]
    values: list[int] = []
    for col in cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        for v in series:
            values.append(int(v))
            if len(values) >= max_values:
                return values
    return values


def write_test_file(out_path: Path, values: list[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for v in values:
            f.write(f'{v}\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-values', type=int, default=DEFAULT_INTEGER_MAX_VALUES)
    parser.add_argument('--include-all', action='store_true', help='Include files outside INTEGER_DATASET_MAPPING')
    parser.add_argument(
        '--paper-table',
        action='store_true',
        help='Only the 13 datasets in paper Table dataset (fig12/fig13)',
    )
    args = parser.parse_args()

    if not INTEGER_SRC.is_dir():
        raise SystemExit(f'Missing integer_dataset: {INTEGER_SRC}')

    if args.paper_table:
        targets = sorted(PAPER_TABLE_DATASET_FILES)
        col_map = INTEGER_DATASET_COLUMNS
    elif args.include_all:
        targets = ALL_INTEGER_DATASET_MAPPING.keys()
        col_map = ALL_INTEGER_DATASET_COLUMNS
    else:
        targets = INTEGER_DATASET_MAPPING.keys()
        col_map = INTEGER_DATASET_COLUMNS
    if TEST_DATA.is_dir():
        for old in TEST_DATA.iterdir():
            if old.name.endswith('.csv'):
                old.unlink()
    written = 0
    for name in sorted(targets):
        src = INTEGER_SRC / name
        if not src.is_file():
            print(f'skip {name} (not found)')
            continue
        cols = col_map.get(name)
        limit = max_values_for(name, args.max_values)
        try:
            values = extract_values(src, cols, limit)
        except Exception as e:
            print(f'skip {name}: {e}')
            continue
        if len(values) < 1024:
            print(f'skip {name}: only {len(values)} values')
            continue
        write_test_file(TEST_DATA / name, values)
        print(f'wrote {name}: {len(values)} values')
        written += 1
    print(f'Done: {written} files in {TEST_DATA}')


if __name__ == '__main__':
    main()
