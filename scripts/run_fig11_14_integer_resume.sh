#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

need_dir() {
  local dir="$1" min="${2:-12}"
  [ -d "results/$dir" ] && [ "$(find "results/$dir" -maxdepth 1 -name '*__*.csv' | wc -l)" -ge "$min" ]
}

echo "==> Ensure TestData (integer_dataset)"
python3 scripts/prepare_integer_testdata.py

if ! need_dir output_BP 12 || ! need_dir output_BP_only_Prune_Plus_RMQ_all_no8 12 || ! need_dir output_Sprintz_only_Prune_Plus_RMQ_all_no8 12; then
  echo "==> Fig11 benchmarks"
  mvn -q test '-Dtest=OptimizePackSizeImpoveCompressionRatio#BP+BPPruneRMQ+SprintzPruneRMQ+Sprintz'
  mvn -q test '-Dtest=OptimizePackSizeVaryPackSize#BP+Sprintz'
else
  echo "==> Skip Fig11 benchmarks (results present)"
fi

if ! need_dir output_BP_vary_pack_size 12; then
  echo "==> Fig12 benchmarks"
  mvn -q test '-Dtest=OptimizePackSizeVaryPackSize#BPPruneRMQ+BPPrune+BPAll+SprintzPruneRMQ+SprintzAll+SprintzPrune+BP+Sprintz'
else
  echo "==> Skip Fig12 benchmarks (results present)"
fi

if ! need_dir output_BP_vary_page_size 12; then
  echo "==> Fig13 benchmarks (VaryPageSize)"
  mvn -q test '-Dtest=OptimizePackSizeVaryPageSize#BP+Sprintz+BPAll+SprintzAll+BPonlyPrune+SprintzonlyPrune+BPPruneRMQ+SprintzPruneRMQ+VaryPageSizeOptimizePackSizeFiltersPlus+VaryPageSizeOptimizePackSizeFiltersPlusSprintz'
else
  echo "==> Skip Fig13 benchmarks (results present)"
fi

if ! need_dir output_BP_filters_plus 12; then
  echo "==> Fig14 benchmarks (Filter)"
  mvn -q test '-Dtest=OptimizePackSizeFilter#BPPruneRMQ+SprintzPruneRMQ'
else
  echo "==> Skip Fig14 benchmarks (results present)"
fi

echo "==> Plot fig11–fig14"
mkdir -p figure_for_paper
python3 fig_combine_results.py
python3 fig11_improve_compare_ratio.py
python3 fig12_vary_pack_size.py
python3 fig13_vary_page_size.py
python3 fig14_fileter_p_prune_plus.py

echo "Done. Figures under $ROOT/figure_for_paper/"
