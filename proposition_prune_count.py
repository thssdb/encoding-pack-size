"""Count proposition-based pack-size prunes (V6+ RMQ), mirroring OptimizePackSizeVaryPackSize.

Counts every prune from optimizePackSizeallV6PlusProposition except skips after a full
pack-cost evaluation where currentCost > cost[prev] (direct cost monotonicity).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

ICDE_ROOT = Path(__file__).resolve().parent
_FILTER_JAVA = ICDE_ROOT / 'src' / 'encoding' / 'packsize' / 'OptimizePackSizeFilter.java'
CHUNK_SIZE = 1024


def _load_prev_array() -> list[int]:
    text = _FILTER_JAVA.read_text(encoding='utf-8')
    m = re.search(r'PREV_ARRAY = \{([^}]+)\}', text, re.DOTALL)
    if not m:
        raise RuntimeError('PREV_ARRAY not found in OptimizePackSizeFilter.java')
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    prev = [0] * (len(nums) + 1)
    for i, v in enumerate(nums):
        prev[i] = v
    return prev


PREV_ARRAY = _load_prev_array()


def sprintz(values: list[int]) -> list[int]:
    if not values:
        return []
    out = [values[0]]
    prev = values[0]
    for i in range(1, len(values)):
        current = values[i]
        diff = current - prev
        encoded = (diff << 1) ^ (diff >> 63)
        out.append(encoded)
        prev = current
    return out


def _bit_width(value: int) -> int:
    v = max(1, value)
    return v.bit_length()


def _build_log2(n: int) -> list[int]:
    log2 = [0] * (n + 1)
    for i in range(2, n + 1):
        log2[i] = log2[i // 2] + 1
    return log2


def _build_sparse_table(bit_widths: list[int]) -> list[list[int]]:
    n = len(bit_widths)
    log_n = max(1, n).bit_length()
    st = [[0] * n for _ in range(log_n)]
    st[0] = bit_widths[:]
    for k in range(1, log_n):
        step = 1 << (k - 1)
        for i in range(n - (1 << k) + 1):
            st[k][i] = max(st[k - 1][i], st[k - 1][i + step])
    return st


def _rmq_max(st: list[list[int]], log2: list[int], start: int, end: int) -> int:
    if start > end:
        return 0
    length = end - start + 1
    k = log2[length]
    return max(st[k][start], st[k][end - (1 << k) + 1])


def _range_max(y: list[int], start: int, end: int) -> int:
    return max(y[start : end + 1]) if start <= end else 0


def _v6_plus_non_decreasing_lower_bound(
    s: int, prefix_max: list[int], suffix_sum_after_s: list[int], z: int
) -> int:
    return s * prefix_max[s] + suffix_sum_after_s[s] + z


def _v6_plus_non_increasing_lower_bound(s: int, n: int, sum_bit_widths: int, z: int) -> int:
    return sum_bit_widths + ((n + s - 1) // s) * z


def _v6_plus_combined_lower_bound(
    p: int,
    n: int,
    prefix_max: list[int],
    suffix_sum_after_s: list[int],
    sum_bit_widths: int,
    z: int,
) -> int:
    return max(
        _v6_plus_non_decreasing_lower_bound(p, prefix_max, suffix_sum_after_s, z),
        _v6_plus_non_increasing_lower_bound(p, n, sum_bit_widths, z),
    )


def _v6_plus_non_decreasing_cost_region(s_ref: int, n: int, y: list[int]) -> bool:
    if s_ref < (n + 1) // 2 or s_ref >= n:
        return False
    max_mid = _range_max(y, s_ref, n - 2)
    yn = y[n - 1]
    max_prefix = _range_max(y, 0, s_ref - 1)
    return max_mid <= yn <= max_prefix


def _precompute_bounds(bit_widths: list[int], n: int):
    prefix_max = [0] * (n + 1)
    suffix_sum_after_s = [0] * (n + 1)
    running_max = 0
    total = 0
    for s in range(1, n + 1):
        total += bit_widths[s - 1]
        if bit_widths[s - 1] > running_max:
            running_max = bit_widths[s - 1]
        prefix_max[s] = running_max
    for s in range(n - 1, 0, -1):
        suffix_sum_after_s[s] = bit_widths[s] + suffix_sum_after_s[s + 1]
    return prefix_max, suffix_sum_after_s, total


def _compute_pack_cost_rmq(
    p: int, n: int, st: list[list[int]], log2: list[int], z: int
) -> int:
    m = (n + p - 1) // p
    r = n - (m - 1) * p
    current_cost = 0
    for i in range(m - 1):
        start = i * p
        end = start + p - 1
        current_cost += p * _rmq_max(st, log2, start, end)
    if m > 0 and r > 0:
        last_start = (m - 1) * p
        current_cost += r * _rmq_max(st, log2, last_start, n - 1)
    return current_cost + m * z


def _compute_pack_cost_n1_rmq(st: list[list[int]], log2: list[int], n: int, bit_widths: list[int], z: int) -> int:
    return (n - 1) * _rmq_max(st, log2, 0, n - 2) + bit_widths[n - 1] + 2 * z


def _skip_by_lower_bound(
    p: int,
    prev: int,
    cost_lb: int,
    best_cost: int,
    cost: list[int],
    is_increased: list[bool],
) -> bool:
    if cost_lb < best_cost:
        return False
    cost[p] = cost_lb
    if prev != 0 and cost_lb > cost[prev]:
        is_increased[p] = True
    return True


def count_proposition_prunes(values: list[int], *, use_rmq: bool = True) -> int:
    """Return number of pruned pack sizes for one chunk (length n)."""
    n = len(values)
    if n < 8:
        return 0

    bit_widths = [0] * n
    global_max = 0
    is_b_n_b_max = False
    has_larger_b_n = False
    bound_index = n - 1
    half_n = n // 2
    last_value = values[n - 1]

    for i in range(n - 1, half_n - 1, -1):
        value = values[i]
        if value > global_max:
            global_max = value
        if not has_larger_b_n and value > last_value:
            has_larger_b_n = True
            bound_index = i
        bit_widths[i] = _bit_width(value)
    if global_max == values[n - 1]:
        is_b_n_b_max = True

    for i in range(half_n):
        value = values[i]
        if value > global_max:
            global_max = value
        bit_widths[i] = _bit_width(value)

    prefix_max, suffix_sum_after_s, sum_bit_widths = _precompute_bounds(bit_widths, n)

    st: list[list[int]] | None = None
    log2: list[int] | None = None
    if use_rmq:
        st = _build_sparse_table(bit_widths)
        log2 = _build_log2(n)

    bit_width_global = _bit_width(global_max)
    z = int(math.ceil(math.log(bit_width_global + 1) / math.log(2)))

    cost = [0] * (n + 1)
    is_increased = [False] * (n + 1)
    best_pack_size = n
    best_cost = n * bit_width_global + z
    prune_count = 0

    non_decreasing_cost_region = _v6_plus_non_decreasing_cost_region(half_n, n, bit_widths)

    def full_cost(p: int) -> int:
        assert st is not None and log2 is not None
        return _compute_pack_cost_rmq(p, n, st, log2, z)

    for p in range(1, half_n + 1):
        prev = PREV_ARRAY[p] if p < len(PREV_ARRAY) else 0
        if prev != 0 and is_increased[prev]:
            is_increased[p] = True
            prune_count += 1
            continue

        cost_lb = _v6_plus_combined_lower_bound(
            p, n, prefix_max, suffix_sum_after_s, sum_bit_widths, z
        )
        if _skip_by_lower_bound(p, prev, cost_lb, best_cost, cost, is_increased):
            prune_count += 1
            continue

        current_cost = full_cost(p)
        cost[p] = current_cost
        if prev != 0 and current_cost > cost[prev]:
            is_increased[p] = True
            continue

        if current_cost < best_cost:
            best_cost = current_cost
            best_pack_size = p

    if not is_b_n_b_max:
        cost_lb = _v6_plus_combined_lower_bound(
            n - 1, n, prefix_max, suffix_sum_after_s, sum_bit_widths, z
        )
        if cost_lb >= best_cost:
            prune_count += 1
        elif st is not None and log2 is not None:
            current_cost = _compute_pack_cost_n1_rmq(st, log2, n, bit_widths, z)
            if current_cost < best_cost:
                best_cost = current_cost
                best_pack_size = n - 1

    for p in range(half_n + 1, bound_index + 1):
        prev = PREV_ARRAY[p] if p < len(PREV_ARRAY) else 0
        if prev != 0 and is_increased[prev]:
            is_increased[p] = True
            prune_count += 1
            continue

        cost_lb = _v6_plus_combined_lower_bound(
            p, n, prefix_max, suffix_sum_after_s, sum_bit_widths, z
        )
        if _skip_by_lower_bound(p, prev, cost_lb, best_cost, cost, is_increased):
            prune_count += 1
            if non_decreasing_cost_region:
                prune_count += bound_index - p
                break
            continue

        current_cost = full_cost(p)
        cost[p] = current_cost
        if prev != 0 and current_cost > cost[prev]:
            is_increased[p] = True
            if non_decreasing_cost_region:
                prune_count += bound_index - p
                break
            continue

        if current_cost < best_cost:
            best_cost = current_cost
            best_pack_size = p

        if non_decreasing_cost_region and current_cost >= best_cost:
            prune_count += bound_index - p
            break

    for p in range(bound_index + 1, n - 1):
        prune_count += 1

    return prune_count


def scale_integers(values: list[int]) -> list[int]:
    if not values:
        return []
    m = min(values)
    return [v - m for v in values]


def load_scaled_values(path: Path) -> list[int]:
    raw: list[int] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw.append(int(float(s)))
    return scale_integers(raw)


def chunks_of_page_size(values: list[int], page_size: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    for i in range(0, len(values), page_size):
        chunk = values[i : i + page_size]
        if len(chunk) >= 8:
            chunks.append(chunk)
    return chunks


def load_testdata_chunks(path: Path, chunk_size: int = CHUNK_SIZE) -> list[list[int]]:
    return chunks_of_page_size(load_scaled_values(path), chunk_size)


def mean_prune_pct_for_file(path: Path, *, sprintz_encode: bool = False) -> float | None:
    chunks = load_testdata_chunks(path)
    if not chunks:
        return None
    counts: list[float] = []
    for chunk in chunks:
        data = sprintz(chunk) if sprintz_encode else chunk
        counts.append(count_proposition_prunes(data))
    mean_count = sum(counts) / len(counts)
    return mean_count / CHUNK_SIZE * 100.0


def mean_prune_pct_for_page_size(
    path: Path, page_size: int, *, sprintz_encode: bool = False
) -> float | None:
    """Proposition prune rate (%) for one dataset at page size n (same rule as fig14, denominator n)."""
    chunks = chunks_of_page_size(load_scaled_values(path), page_size)
    if not chunks:
        return None
    counts: list[float] = []
    for chunk in chunks:
        data = sprintz(chunk) if sprintz_encode else chunk
        counts.append(float(count_proposition_prunes(data)))
    mean_count = sum(counts) / len(counts)
    return mean_count / page_size * 100.0


def build_vary_page_prune_rate_data(
    testdata_dir: Path,
    page_sizes: list[int],
    *,
    dataset_names: set[str] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Pruning rate panels for fig13: BP-Prune and Sprintz-Prune by page size."""
    out: dict[str, dict[int, dict[str, float]]] = {
        'BP-Prune': {n: {} for n in page_sizes},
        'Sprintz-Prune': {n: {} for n in page_sizes},
    }
    if not testdata_dir.is_dir():
        return out
    for path in sorted(testdata_dir.glob('*.csv')):
        name = path.name
        if dataset_names is not None and name not in dataset_names:
            continue
        for page_size in page_sizes:
            bp_pct = mean_prune_pct_for_page_size(path, page_size, sprintz_encode=False)
            if bp_pct is not None:
                out['BP-Prune'][page_size][name] = bp_pct
            sp_pct = mean_prune_pct_for_page_size(path, page_size, sprintz_encode=True)
            if sp_pct is not None:
                out['Sprintz-Prune'][page_size][name] = sp_pct
    return out


def collect_proposition_prune_pcts(
    testdata_dir: Path,
    *,
    sprintz_encode: bool = False,
    dataset_names: set[str] | None = None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not testdata_dir.is_dir():
        return results
    for path in sorted(testdata_dir.glob('*.csv')):
        name = path.name
        if dataset_names is not None and name not in dataset_names:
            continue
        pct = mean_prune_pct_for_file(path, sprintz_encode=sprintz_encode)
        if pct is not None:
            results[path.stem] = pct
    return results
