#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate

need_dir() {
  local dir="$1" min="${2:-17}"
  [ -d "results/$dir" ] && [ "$(find "results/$dir" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')" -ge "$min" ]
}

run_vary_page() {
  local method="$1" outdir="$2"
  if need_dir "$outdir"; then
    echo "skip VaryPageSize#$method ($outdir done)"
    return 0
  fi
  echo "==> mvn VaryPageSize#$method -> results/$outdir"
  mvn -q test "-Dtest=OptimizePackSizeVaryPageSize#$method"
}

# Fig13: only missing output directories
run_vary_page BP output_BP_vary_page_size
run_vary_page Sprintz output_sprintz_vary_page_size
run_vary_page BPAll output_BP_vary_page_size_N2
run_vary_page SprintzAll output_Sprintz_vary_page_size_N2
run_vary_page SprintzonlyPrune output_Sprintz_only_Prune_vary_page_size
run_vary_page BPPruneRMQ output_BP_Prune_RMQ_vary_page_size
run_vary_page SprintzPruneRMQ output_Sprintz_Prune_RMQ_vary_page_size
run_vary_page VaryPageSizeOptimizePackSizeFiltersPlus output_BP_filters_plus_vary_page_size
run_vary_page VaryPageSizeOptimizePackSizeFiltersPlusSprintz output_Sprintz_filters_plus_vary_page_size

if ! need_dir output_BP_filters_plus; then
  echo "==> Fig14 Filter"
  mvn -q test '-Dtest=OptimizePackSizeFilter#BPPruneRMQ+SprintzPruneRMQ'
else
  echo "skip Fig14 (filters_plus done)"
fi

echo "==> Plot fig11–fig14"
mkdir -p figure_for_paper
python3 fig_combine_results.py
python3 fig11_improve_compare_ratio.py
python3 fig12_vary_pack_size.py
python3 fig13_vary_page_size.py
python3 fig14_fileter_p_prune_plus.py
echo "Done."
