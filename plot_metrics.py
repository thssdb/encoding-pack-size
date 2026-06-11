"""Convert benchmark CSV time columns to plot values (ns/point)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Java writes throughput (MB/s) in Encoding/Decoding Time for these result folders.
_THROUGHPUT_DIR_NAMES = frozenset(
    {
        'output_BP_vary_pack_size',
        'output_Sprintz_vary_pack_size',
        'output_BP_All_vary_pack_size',
        'output_BP_all_no8',
        'output_BP_only_Prune_Plus_all_no8',
        'output_BP_only_Prune_Plus_RMQ_all_no8',
        'output_Sprintz_only_Prune_Plus_all_no8',
        'output_Sprintz_only_Prune_Plus_RMQ_all_no8',
        'output_Sprintz_N2_all_no8',
        'output_BP_Prune_all_no8',
        'output_Sprintz_Prune_all_no8',
        'output_BP_RMQ_all_no8_sprintz',
        'output_BP_vary_page_size',
        'output_BP_vary_page_size_N2',
        'output_BP_only_Prune_vary_page_size',
        'output_BP_Prune_RMQ_vary_page_size',
        'output_sprintz_vary_page_size',
        'output_Sprintz_vary_page_size_N2',
        'output_Sprintz_only_Prune_vary_page_size',
        'output_Sprintz_Prune_RMQ_vary_page_size',
    }
)

# Legacy runs store ns/point directly (values typically < 50).
_NS_PER_POINT_DIR_NAMES = frozenset()


def time_column_is_throughput(result_dir: Path | str | None) -> bool | None:
    """Return True/False if known; None → use magnitude heuristic."""
    if result_dir is None:
        return None
    name = result_dir.name if isinstance(result_dir, Path) else Path(str(result_dir)).name
    if name in _THROUGHPUT_DIR_NAMES:
        return True
    if name in _NS_PER_POINT_DIR_NAMES:
        return False
    return None


def csv_time_to_ns_per_point(t: float, *, result_dir: Path | str | None = None) -> float:
    """Map CSV time column to ns/point for fig12/13."""
    if not np.isfinite(t) or t <= 0:
        return np.nan
    mode = time_column_is_throughput(result_dir)
    if mode is True:
        return 8000.0 / t
    if mode is False:
        return float(t)
    if t > 50:
        return 8000.0 / t
    return float(t)
