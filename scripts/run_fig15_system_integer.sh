#!/usr/bin/env bash
# Fig. 15 (system compare): integer_dataset -> C++ benchmark -> fig15_system.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENCODING_ROOT="$(cd "$ROOT/.." && pwd)"

if [ -d "$ROOT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

resolve_tsfile_cpp() {
  if [ -n "${TSFILE_CPP_DIR:-}" ] && [ -d "$TSFILE_CPP_DIR" ]; then
    echo "$TSFILE_CPP_DIR"
    return
  fi
  local candidates=(
    "$ENCODING_ROOT/../tsfile/cpp"
    "$ROOT/tsfile/cpp"
    "$HOME/Documents/GitHub/tsfile/cpp"
  )
  local d
  for d in "${candidates[@]}"; do
    if [ -d "$d" ]; then
      echo "$(cd "$d" && pwd)"
      return
    fi
  done
  echo ""
}

resolve_test_binary() {
  local cpp_root="$1"
  if [ -n "${TSFILE_TEST_BIN:-}" ] && [ -x "$TSFILE_TEST_BIN" ]; then
    echo "$TSFILE_TEST_BIN"
    return
  fi
  local cand
  for cand in \
    "$cpp_root/build/test/lib/TsFile_Test" \
    "$cpp_root/build/lib/TsFile_Test" \
    "$cpp_root/build/Release/test/lib/TsFile_Test"; do
    if [ -x "$cand" ]; then
      echo "$cand"
      return
    fi
  done
  echo ""
}

echo "==> Prepare integer TestData"
cd "$ROOT"
python3 scripts/prepare_integer_testdata.py

TSFILE_CPP="$(resolve_tsfile_cpp)"
if [ -z "$TSFILE_CPP" ]; then
  echo "error: tsfile cpp tree not found. Set TSFILE_CPP_DIR to .../tsfile/cpp" >&2
  exit 1
fi
echo "==> Using tsfile cpp: $TSFILE_CPP"

configure_tsfile_cpp() {
  local cpp_root="$1"
  local build_dir="$cpp_root/build"
  local sdkroot
  sdkroot="$(xcrun --show-sdk-path 2>/dev/null || true)"
  mkdir -p "$build_dir"
  cmake -S "$cpp_root" -B "$build_dir" \
    -DBUILD_TEST=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    ${sdkroot:+-DCMAKE_OSX_SYSROOT="$sdkroot"} \
    ${sdkroot:+-DCMAKE_C_FLAGS="-isysroot $sdkroot"} \
    ${sdkroot:+-DCMAKE_CXX_FLAGS="-isysroot $sdkroot -Wall -std=c++11"} \
    -DENABLE_ANTLR4=OFF \
    -DENABLE_SNAPPY=ON -DENABLE_LZ4=ON -DENABLE_LZOKAY=ON -DENABLE_ZLIB=ON
}

TEST_BIN="$(resolve_test_binary "$TSFILE_CPP")"
if [ -z "$TEST_BIN" ]; then
  echo "==> Configure + build TsFile_Test"
  configure_tsfile_cpp "$TSFILE_CPP"
  cmake --build "$TSFILE_CPP/build" --target TsFile_Test -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
  TEST_BIN="$(resolve_test_binary "$TSFILE_CPP")"
fi
if [ -z "$TEST_BIN" ]; then
  echo "error: TsFile_Test binary not found after build" >&2
  exit 1
fi
echo "==> Using test binary: $TEST_BIN"

export TSFILE_BENCHMARK_DATA_DIR="$ROOT/TestData"
export TSFILE_BENCHMARK_WARMUP="${TSFILE_BENCHMARK_WARMUP:-0}"
export TSFILE_BENCHMARK_MEASURE_REPEATS="${TSFILE_BENCHMARK_MEASURE_REPEATS:-5}"
# Optional smoke test: TSFILE_DATASET_LIMIT=2 ./scripts/run_fig15_system_integer.sh

echo "==> C++ benchmark (SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder)"
"$TEST_BIN" --gtest_filter=SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder

echo "==> Plot fig15_system.py"
cd "$ROOT"
python3 fig15_system.py

echo "Done."
echo "  CSV:  $ENCODING_ROOT/output_tsfile_packsize_comparison_cpp/tsfile_comparison_cpp.csv"
echo "  Figure: $ENCODING_ROOT/figure_for_paper/system_compare.png"
