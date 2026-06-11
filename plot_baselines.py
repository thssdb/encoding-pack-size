"""Load per-dataset baseline metrics from Java benchmark CSV outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from integer_datasets import INTEGER_DATASET_MAPPING, RESULTS_DIR
from plot_metrics import csv_time_to_ns_per_point

# Prune-RMQ = Prune (V6Plus) + RMQ; CR matches Prune at optimal pack.
_BP_PRUNE = RESULTS_DIR / 'output_BP_only_Prune_Plus_all_no8'
_BP_PRUNE_RMQ = RESULTS_DIR / 'output_BP_only_Prune_Plus_RMQ_all_no8'
_BP_ALL_OPT = RESULTS_DIR / 'output_BP_all_no8'
_SZ_PRUNE = RESULTS_DIR / 'output_Sprintz_only_Prune_Plus_all_no8'
_SZ_PRUNE_RMQ = RESULTS_DIR / 'output_Sprintz_only_Prune_Plus_RMQ_all_no8'
_SZ_ALL_OPT = RESULTS_DIR / 'output_Sprintz_N2_all_no8'
_BP_VARY = RESULTS_DIR / 'output_BP_vary_pack_size'
_SZ_VARY = RESULTS_DIR / 'output_Sprintz_vary_pack_size'

FIG12_BASELINE_DIRS = {
    'bp': {
        'prune_rmq': _BP_PRUNE_RMQ,
        'prune': _BP_PRUNE,
        'all': _BP_ALL_OPT,
    },
    'sz': {
        'prune_rmq': _SZ_PRUNE_RMQ,
        'prune': _SZ_PRUNE,
        'all': _SZ_ALL_OPT,
    },
}


def _metrics_from_row(row, *, result_dir: Path | None = None) -> tuple[float, float, float]:
    cr = float(row['Compression Ratio'])
    enc = float(row['Encoding Time'])
    dec = float(row['Decoding Time'])
    inv_cr = (1.0 / cr) if cr and np.isfinite(cr) and cr > 0 else np.nan
    inv_enc = csv_time_to_ns_per_point(enc, result_dir=result_dir)
    inv_dec = csv_time_to_ns_per_point(dec, result_dir=result_dir)
    return inv_cr, inv_enc, inv_dec


def _baseline_csv_files(
    result_dir: Path,
    dataset_mapping: dict[str, str] | None = None,
    *,
    dataset_names: set[str] | frozenset[str] | None = None,
) -> list[Path]:
    """CSV paths to load: prefer mapping keys present on disk, else all result CSVs."""
    dataset_mapping = dataset_mapping or INTEGER_DATASET_MAPPING
    mapped = [
        result_dir / fname
        for fname in sorted(dataset_mapping.keys())
        if (result_dir / fname).is_file() and (dataset_names is None or fname in dataset_names)
    ]
    if mapped:
        return mapped
    if dataset_names is not None:
        return []
    return sorted(result_dir.glob('*.csv'))


def load_single_run_baseline(
    result_dir: Path,
    dataset_mapping: dict[str, str] | None = None,
    *,
    dataset_names: set[str] | frozenset[str] | None = None,
):
    """One row per dataset CSV (fixed optimal pack size)."""
    cr_l, enc_l, dec_l = [], [], []
    if not result_dir.is_dir():
        return (np.array([]), np.array([]), np.array([]))
    for path in _baseline_csv_files(result_dir, dataset_mapping, dataset_names=dataset_names):
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            icr, ie, id_ = _metrics_from_row(df.iloc[0], result_dir=result_dir)
            if np.isfinite(icr):
                cr_l.append(icr)
            if np.isfinite(ie):
                enc_l.append(ie)
            if np.isfinite(id_):
                dec_l.append(id_)
        except Exception:
            continue
    return (np.asarray(cr_l, dtype=float), np.asarray(enc_l, dtype=float), np.asarray(dec_l, dtype=float))


def load_best_vary_pack_baseline(
    vary_dir: Path,
    dataset_mapping: dict[str, str] | None = None,
    *,
    dataset_names: set[str] | frozenset[str] | None = None,
):
    """Best compression ratio (max 1/CR) across pack sizes; enc/dec at that pack size."""
    cr_l, enc_l, dec_l = [], [], []
    if not vary_dir.is_dir():
        return (np.array([]), np.array([]), np.array([]))
    for path in _baseline_csv_files(vary_dir, dataset_mapping, dataset_names=dataset_names):
        try:
            df = pd.read_csv(path)
            if df.empty or 'Compression Ratio' not in df.columns:
                continue
            best_icr = -np.inf
            best_ie = np.nan
            best_id = np.nan
            for _, row in df.iterrows():
                icr, ie, id_ = _metrics_from_row(row, result_dir=vary_dir)
                if np.isfinite(icr) and icr > best_icr:
                    best_icr = icr
                    best_ie = ie
                    best_id = id_
            if np.isfinite(best_icr) and best_icr > -np.inf:
                cr_l.append(best_icr)
                if np.isfinite(best_ie):
                    enc_l.append(best_ie)
                if np.isfinite(best_id):
                    dec_l.append(best_id)
        except Exception:
            continue
    return (np.asarray(cr_l, dtype=float), np.asarray(enc_l, dtype=float), np.asarray(dec_l, dtype=float))


def populate_fig12_baseline_boxes(
    box_cr: dict,
    box_enc: dict,
    box_dec: dict,
    dirs: dict | None = None,
    *,
    dataset_names: set[str] | frozenset[str] | None = None,
) -> None:
    """Load fig12 optimal-pack baseline bars.

    CR / encode come from single-run optimal-pack benchmarks.
    Decode uses the vary-pack CSV row with best CR so timing matches the
    BP()/Sprintz() curve harness (per-chunk timing), not benchChunkedBitPacking.
    """
    dirs = dirs or FIG12_BASELINE_DIRS
    vary_dec = {
        'bp': load_best_vary_pack_baseline(_BP_VARY, dataset_names=dataset_names)[2],
        'sz': load_best_vary_pack_baseline(_SZ_VARY, dataset_names=dataset_names)[2],
    }
    for family, keys in dirs.items():
        box_cr.setdefault(family, {})
        box_enc.setdefault(family, {})
        box_dec.setdefault(family, {})
        for key, path in keys.items():
            names = dataset_names if key == 'prune_rmq' else None
            cr, enc, _ = load_single_run_baseline(path, dataset_names=names)
            dec = vary_dec.get(family, np.array([]))
            if cr.size:
                box_cr[family][key] = cr
                print(f'fig12 baseline {family}/{key}: CR n={cr.size} mean={np.mean(cr):.4f}')
            if enc.size:
                box_enc[family][key] = enc
                print(f'fig12 baseline {family}/{key}: enc ns/pt mean={np.mean(enc):.2f}')
            if dec.size:
                box_dec[family][key] = dec
                print(f'fig12 baseline {family}/{key}: dec ns/pt mean={np.mean(dec):.2f}')
        # Optimal pack (and CR) is identical for All / Prune / Prune-RMQ; only search cost differs.
        fam_cr = box_cr.get(family, {})
        cr_rmq = fam_cr.get('prune_rmq')
        cr_prune = fam_cr.get('prune')
        cr_all = fam_cr.get('all')
        if cr_rmq is not None and np.size(cr_rmq) > 0:
            canonical = np.asarray(cr_rmq, dtype=float).copy()
        elif cr_prune is not None and np.size(cr_prune) > 0:
            canonical = np.asarray(cr_prune, dtype=float).copy()
        elif cr_all is not None and np.size(cr_all) > 0:
            canonical = np.asarray(cr_all, dtype=float).copy()
        else:
            canonical = None
        if canonical is not None:
            for key in ('prune_rmq', 'prune', 'all'):
                if key in keys:
                    fam_cr[key] = canonical.copy()
            print(
                f'fig12 baseline {family}: CR aligned for all/prune/prune_rmq '
                f'(n={canonical.size}, mean={np.mean(canonical):.4f})'
            )


def algorithms_with_page_data(compression_ratio_data: dict, vector_sizes: list[int]) -> list[str]:
    """Algorithms that have at least one dataset for every page size slot."""
    ok = []
    for algo, by_size in compression_ratio_data.items():
        if all(by_size.get(s) for s in vector_sizes):
            ok.append(algo)
    return ok
