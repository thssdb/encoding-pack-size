"""Integer benchmark datasets (from ../integer_dataset) used by fig11–fig14."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ICDE_ROOT = Path(__file__).resolve().parent
INTEGER_SRC = REPO_ROOT / 'integer_dataset'
RESULTS_DIR = ICDE_ROOT / 'results'
FIGURE_DIR = ICDE_ROOT / 'figure_for_paper'

# Paper Table~\ref{table:dataset} abbreviations (sorted lexicographically by Abbr.).
INTEGER_FILE_ABBREV: dict[str, str] = {
    'adult__Capital_Loss.csv': 'ADC',
    'adult__Education_Num.csv': 'ADE',
    'books.csv': 'BB',
    'bank_train__campaign.csv': 'BKC',
    'bank_train__pdays.csv': 'BKP',
    'credit_scoring__client_id.csv': 'CC',
    'Census-Population.csv': 'CP',
    'ehresp_2014__tucaseid.csv': 'ET',
    'fb.csv': 'FB',
    'MeteoNet-Weather.csv': 'MW',
    'Rail_Insurance_Claims__MILES.csv': 'RM',
    'Rail_Insurance_Claims__WEIGHT.csv': 'RW',
    'scores__User_Count.csv': 'SCUC',
    'video_games_sales__User_Count.csv': 'VGSUC',
    'vgsales__Rank.csv': 'VR',
    'weights_heights__Index.csv': 'WI',
    'wiki_ts.csv': 'WK',
    'EPM-Education.csv': 'EPME',
    'Wine-Tasting.csv': 'WNT',
}

SKIP_INTEGER_FILES = frozenset({
    'EPM-Education.csv',
    'Wine-Tasting.csv',
    'MeteoNet-Weather.csv',
    'wiki_ts.csv',
})

# Large files: sample at most this many points for benchmarks/plots.
LARGE_INTEGER_MAX_VALUES: dict[str, int] = {
    'books.csv': 1_000_000,
    'fb.csv': 1_000_000,
    'Census-Population.csv': 1_000_000,
}
DEFAULT_INTEGER_MAX_VALUES = 80_000

# Single-column files without a header row (one integer per line).
HEADERLESS_INTEGER_FILES = frozenset({
    'books.csv',
    'fb.csv',
    'Census-Population.csv',
})

_SKIP_SCRIPTS = frozenset({
    'split_columns_to_csvs.py',
    'clean_and_merge_csvs.py',
    'remove_non_integer_csvs.py',
    'remove_negative_csvs.py',
    'remove_empty_rows.py',
    'restore_and_filter_integer_csvs.py',
    'split_columns_to_csvs.py',
})


def _file_label(filename: str) -> str:
    """Fallback label when filename is not in INTEGER_FILE_ABBREV."""
    stem = filename.removesuffix('.csv')
    if '__' in stem:
        base, col = stem.split('__', 1)
        col_short = re.sub(r'[^A-Za-z0-9]+', '', col)[:6]
        prefix = re.sub(r'[^A-Za-z0-9]+', '', base)[:4].upper()
        label = f'{prefix}{col_short}' if col_short else prefix
    else:
        label = re.sub(r'[^A-Za-z0-9]+', '', stem)[:8].upper()
    return label[:16] if label else stem[:16]


def discover_integer_datasets() -> tuple[dict[str, str], dict[str, list[int]]]:
    """Each integer_dataset/*.csv is one dataset; column index is always [0]."""
    mapping: dict[str, str] = {}
    columns: dict[str, list[int]] = {}
    used_labels: dict[str, int] = {}

    if not INTEGER_SRC.is_dir():
        return mapping, columns

    for path in sorted(INTEGER_SRC.glob('*.csv')):
        name = path.name
        if name in SKIP_INTEGER_FILES or name in _SKIP_SCRIPTS:
            continue
        if name.endswith('.tmp.csv'):
            continue

        label = INTEGER_FILE_ABBREV.get(name, _file_label(name))
        if label in used_labels:
            used_labels[label] += 1
            label = f'{label}{used_labels[label]}'[:16]
        else:
            used_labels[label] = 0

        mapping[name] = label
        columns[name] = [0]

    return mapping, columns


INTEGER_DATASET_MAPPING, INTEGER_DATASET_COLUMNS = discover_integer_datasets()

# Paper Table~\ref{table:dataset}: 13 integer benchmarks for fig12/fig13 averages.
PAPER_TABLE_DATASET_ABBREVS = frozenset({
    'ADC', 'ADE', 'BB', 'BKC', 'BKP', 'CC', 'CP', 'ET', 'FB', 'RM', 'RW', 'VR', 'WI',
})
PAPER_TABLE_DATASET_FILES = frozenset(
    fname for fname, abbr in INTEGER_FILE_ABBREV.items() if abbr in PAPER_TABLE_DATASET_ABBREVS
)

# Alias for scripts that pass --include-all
ALL_INTEGER_DATASET_MAPPING = INTEGER_DATASET_MAPPING
ALL_INTEGER_DATASET_COLUMNS = INTEGER_DATASET_COLUMNS


def sort_dataset_abbrevs(abbrevs):
    """Lexicographic order for dataset short names on plot x-axes."""
    return sorted(abbrevs)


def sort_items_by_dataset_label(items, label_fn):
    """Sort (key, value) pairs by display label (dictionary order)."""
    return sorted(items, key=lambda kv: label_fn(kv[0]))
