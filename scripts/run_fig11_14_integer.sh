#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Prepare TestData from integer_dataset"
python3 scripts/prepare_integer_testdata.py

echo "==> Fig11 benchmarks (ImproveCompressionRatio + VaryPackSize BP/Sprintz)"
mvn -q test '-Dtest=OptimizePackSizeImpoveCompressionRatio#BP+BPPruneRMQ+SprintzPruneRMQ+Sprintz'
mvn -q test '-Dtest=OptimizePackSizeVaryPackSize#BP+Sprintz'

echo "==> Fig12 benchmarks (VaryPackSize full)"
mvn -q test '-Dtest=OptimizePackSizeVaryPackSize#BPPruneRMQ+BPPrune+BPAll+SprintzPruneRMQ+SprintzAll+SprintzPrune+BP+Sprintz'

echo "==> Fig13 benchmarks (VaryPageSize)"
mvn -q test '-Dtest=OptimizePackSizeVaryPageSize#BP+Sprintz+BPAll+SprintzAll+BPonlyPrune+SprintzonlyPrune+BPPruneRMQ+SprintzPruneRMQ+VaryPageSizeOptimizePackSizeFiltersPlus+VaryPageSizeOptimizePackSizeFiltersPlusSprintz'

echo "==> Fig14 benchmarks (Filter)"
mvn -q test '-Dtest=OptimizePackSizeFilter#BPPruneRMQ+SprintzPruneRMQ'

echo "==> Plot fig11–fig14"
python3 fig_combine_results.py
python3 fig11_improve_compare_ratio.py
python3 fig12_vary_pack_size.py
python3 fig13_vary_page_size.py
python3 fig14_fileter_p_prune_plus.py

echo "Done. Figures under $ROOT/figure_for_paper/"
