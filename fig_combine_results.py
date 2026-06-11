"""Aggregate Java benchmark CSVs into camel_ratio.xlsx for fig11."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from integer_datasets import INTEGER_DATASET_MAPPING, RESULTS_DIR

dataset_mapping = INTEGER_DATASET_MAPPING


def _compression_ratio_from_csv(csv_path: Path, pack_size: int = 8) -> float | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or 'Compression Ratio' not in df.columns:
        return None
    if 'Pack size' in df.columns:
        sub = df[df['Pack size'] == pack_size]
        if not sub.empty:
            return float(sub['Compression Ratio'].iloc[0])
        return float(df['Compression Ratio'].min())
    return float(df['Compression Ratio'].iloc[0])


def build_camel_ratio(results_dir: Path | None = None, out_path: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or RESULTS_DIR
    out_path = out_path or Path('camel_ratio.xlsx')
    dirs = {
        'BP': results_dir / 'output_BP',
        'BP (Prune-RMQ)': results_dir / 'output_BP_only_Prune_Plus_RMQ_all_no8',
        'Sprintz': results_dir / 'output_Sprintz_vary_pack_size',
        'Sprintz (Prune-RMQ)': results_dir / 'output_Sprintz_only_Prune_Plus_RMQ_all_no8',
    }
    columns = sorted(set(dataset_mapping.values()))
    df = pd.DataFrame(index=list(dirs.keys()), columns=columns, dtype=float)
    for algo, out_dir in dirs.items():
        for fname, abbr in dataset_mapping.items():
            val = _compression_ratio_from_csv(out_dir / fname)
            if val is not None:
                df.at[algo, abbr] = round(val, 3)
    df.to_excel(out_path)
    print(f'Wrote {out_path}')
    return df


def analysis_data():
    return build_camel_ratio()


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    build_camel_ratio()
