# On Optimizing Pack Size for Bit-Packing

Artifacts and Python tooling for the **pack-size optimization** experiments reported in the paper. 
This repository is intended to make **figure reproduction** and **inspection of tabulated metrics** straightforward.

**Fastest path:** **§3 → Quick path**, then **§4 → Quick lookup of script** (copy-paste **Commands** there; use the table below for tests and notes).

**Plotting scripts in this directory:** `fig2_draw_run_lengths.py`, `fig3_cost_vary_pack_size.py`, `fig11_improve_compare_ratio.py`, `fig12_vary_pack_size.py`, `fig13_vary_page_size.py`, `fig14_fileter_p_prune_plus.py`, `fig15_system.py`, and `fig_combine_results.py` (tabular input for `fig11`). One-shot helpers: `scripts/run_fig11_14_integer_resume.sh` (Fig. 11–14), `scripts/run_fig15_system_integer.sh` (Fig. 15).

---

## 1. Core method and dynamic packing (implementation location)

- **In this repository:** analysis and plotting, plus **standalone JUnit benchmarks** under **`src/`** (runnable with **`mvn test`** from this directory; see **§3.2**).
- **Implementation of the proposed method (including dynamic packing):** maintained in the **same experimental codebase** used to generate the csv consumed below (e.g. rows such as `Sprintz (Prune-RMQ)`, `BP (Prune-RMQ)`). 
Script comments point to that workflow (for example, references to a Java-side evaluator in `fig_combine_results.py`).

---

## 2. Repository snapshot (Repository link, branch, and commit)

### The repository repo1 of scripts

Repository link: https://github.com/thssdb/encoding-pack-size

Branch: main

Commit hash: 97043c2b8a590917277810fa941521d8fb3f5f25

### The repository repo2 of the core implementation

Repository link: https://github.com/apache/tsfile/tree/research/encoding-pack-size/

Branch: research/encoding-pack-size

Commit hash: dabbb4176f80f2a3f33a4acdca500ba2296e9d14


---

## 3. Commands to run

### Quick path (reproduce one figure)

Do steps **in this order**; the figure-specific part is **§4**.

1. **One-time setup** — **§3.1** (`./scripts/bootstrap_env.sh` or manual steps: Python venv, **`requirements.txt`**; **tsfile** clone is optional unless you need the upstream Java tree).
2. **Find your script** — open **§4 → Quick lookup of script**, use the **Commands** column for copy-paste steps; open the **detailed table** below for Java test names and notes.
3. **Generate inputs (Java)** — run the numbered **`mvn test …`** lines in **Quick lookup of script** from **`{basedir}`** (this directory; see **§3.2**). Skip `mvn` if that row is Python-only (e.g. `fig15_system.py`).
4. **Plot (Python)** — `cd` into **`{basedir}`**, activate the venv, run the **`python3 …`** lines from the same row. Figures usually write to `./figure_for_paper/` (see **§3.3**).

`{basedir}` is the root of this **`pack-size-for-icde27`** checkout: run both **`mvn test …`** and **`python3 fig*.py`** here. Test sources live under **`src/`**; you do not need a **tsfile** checkout for these JUnit runs unless you are working inside the full TsFile reactor.

### 3.1 Environment

Use **Python 3.9 or newer**; the commands below call the interpreter as **`python3`** (on some Windows installs, use **`py -3`** in the same places). This repo pins **Python 3.11** in **`mise.toml`** / **`.python-version`** for reproducibility; older 3.9+ interpreters still work if you create the venv manually.

**Declared versions (config files)**

| Role | File | Purpose |
|------|------|---------|
| Python packages | **`requirements.txt`** | `pip install -r requirements.txt` inside the venv |
| Python + JDK (optional) | **`mise.toml`** | If you use [mise](https://mise.jdx.dev/), run **`mise install`** in this repo, then open a new shell |
| Python for pyenv | **`.python-version`** | `pyenv` picks 3.11 when you `cd` here |

**Clone repository**

```bash
git clone https://github.com/thssdb/encoding-pack-size.git
cd encoding-pack-size/pack-size-for-icde27
git clone -b research/encoding-pack-size --single-branch https://github.com/apache/tsfile/
```

**One-command setup**

```bash
./scripts/bootstrap_env.sh
```

This creates **`.venv`**, installs dependencies from **`requirements.txt`**, and clones **`tsfile/`** on the `research/encoding-pack-size` branch if missing. It does **not** install the JDK or Maven; install **JDK 17** (or 11+) and **Maven 3.6+** yourself (or via **`mise install`** using **`mise.toml`**), then use **§3.2**.

**Manual setup (equivalent steps)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

After this, **`{basedir}`** is the `pack-size-for-icde27` directory (it may also contain a `tsfile/` subfolder if you ran the clone step above).

### 3.2 Required inputs (Java / Maven)

Benchmark CSVs are produced by **JUnit** in this directory. From **`{basedir}`**, with **JDK 11+** and **Maven 3.6+**:

```bash
cd {basedir}
mvn test '-Dtest=OptimizePackSizeTest#testPackSizeCostAnalysis'
```

Use the exact **`mvn test '-Dtest=…'`** selectors in **§4 → Quick lookup of script**. Join methods on the same class with `+`; separate classes with `,` (see [Maven Surefire](https://maven.apache.org/surefire/maven-surefire-plugin/examples/single-test.html)). **`FloatToIntLosslessTest`** lives on **`FloatToInteger`**: `mvn test '-Dtest=FloatToInteger#FloatToIntLosslessTest'`.

The **tsfile** branch in **§2** remains the canonical home of the same sources for Apache TsFile integration; you only need that checkout if you are hacking or validating inside the full **tsfile** Maven reactor.

### 3.3 Figures (plotting)

From **`{basedir}`**, with the venv activated:

```bash
cd {basedir}
source .venv/bin/activate   # if not already active
python3 <script-from-section-4>.py
```

Use the **`python3 …`** step(s) in **§4 → Quick lookup of script** for your script. Path edits and data prerequisites are in the **Note** column of the detailed table. Outputs are usually under `./figure_for_paper/` as PNG/EPS.

---

## 4. Figure <-> script and test functions mapping

### Quick lookup of script

| Paper figure | Script(s) | Commands (run in order) |
|--------------|-----------|-------------------------|
| 2 | `fig2_draw_run_lengths.py`; `OptimizePackSizeTest.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#FeatureTest+FeatureAfterSprintzTest'`<br>2. `cd {basedir} && python3 fig2_draw_run_lengths.py` |
| 3–4 | `fig3_cost_vary_pack_size.py`; `OptimizePackSizeTest.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#testPackSizeCostAnalysis'`<br>2. `cd {basedir} && python3 fig3_cost_vary_pack_size.py` (edit `csv_dir` / `output_dir` in `if __name__ == '__main__'` if needed; one run produces both Fig. 3 and Fig. 4 outputs.) |
| 11 | `fig11_improve_compare_ratio.py`, `fig_combine_results.py`; `OptimizePackSizeImpoveCompressionRatio.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeImpoveCompressionRatio#BP+BPPruneRMQ+SprintzPruneRMQ+Sprintz'`<br>2. `cd {basedir} && python3 fig_combine_results.py`<br>3. `cd {basedir} && python3 fig11_improve_compare_ratio.py` |
| 12 | `fig12_vary_pack_size.py`; `OptimizePackSizeVaryPackSize.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeVaryPackSize#BPPruneRMQ+BPPrune+BPAll+SprintzPruneRMQ+SprintzAll+SprintzPrune+BP+Sprintz'`<br>2. `cd {basedir} && python3 fig12_vary_pack_size.py` |
| 13 | `fig13_vary_page_size.py`; `OptimizePackSizeVaryPageSize.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeVaryPageSize#BP+Sprintz+BPAll+SprintzAll+BPonlyPrune+SprintzonlyPrune+BPPruneRMQ+SprintzPruneRMQ+VaryPageSizeOptimizePackSizeFiltersPlus+VaryPageSizeOptimizePackSizeFiltersPlusSprintz'`<br>2. `cd {basedir} && python3 fig13_vary_page_size.py` |
| 14 | `fig14_fileter_p_prune_plus.py`; `OptimizePackSizeFilter.java` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeFilter#BPPruneRMQ+SprintzPruneRMQ'`<br>2. `cd {basedir} && python3 fig14_fileter_p_prune_plus.py` |
| 15 | `fig15_system.py`; C++ `SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder` | 1. `cd {basedir} && ./scripts/run_fig15_system_integer.sh`<br>Or manually: prepare `TestData/`, run C++ benchmark, then `python3 fig15_system.py` (see **§4.1**). |

**Surefire:** run from **`{basedir}`** with a quoted **`-Dtest=Class#method…`** selector, as in **§3.2** and the **Commands (run in order)** column above. Set `{basedir}` as in **§3 Quick path** and **§3.1**.

### Detailed mapping (tests and notes)

| Manuscript item | Script | Test function(s) of results | Note |
|-----------------|--------|------------------------------|------|
| Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz | `fig2_draw_run_lengths.py`; `OptimizePackSizeTest.java` | `FeatureTest()` and `FeatureAfterSprintzTest()` in `src/OptimizePackSizeTest.java` (Surefire: `OptimizePackSizeTest#FeatureTest+FeatureAfterSprintzTest`) | Builds combined histograms from `data/features_and_best_p.csv` (BP) and `data/features_and_best_p_sprintz.csv` (Sprintz). |
| Figure 3: Total storage cost of 1024 values of dataset PM10 (Table 2) under various pack sizes 𝑠. It consists of two parts, the bit width cost and the value cost. | `fig3_cost_vary_pack_size.py`; `OptimizePackSizeTest.java` | `testPackSizeCostAnalysis()` in `src/OptimizePackSizeTest.java` (Surefire: `OptimizePackSizeTest#testPackSizeCostAnalysis`) | Figure 3 panels: `fig_of_cost_values_bitwidth_in_chunk(...)`. Edit `csv_dir` / `output_dir` in `if __name__ == '__main__'` if needed. |
| Figure 4: Example on unimodality of the cost function 𝐶(𝑠) on a subset of pack sizes 𝑠 = 3*2^𝛽 , 𝛽 = 0, 1, 2, . . . , according to Proposition 1 | `fig3_cost_vary_pack_size.py`; `OptimizePackSizeTest.java` | `testPackSizeCostAnalysis()` in `src/OptimizePackSizeTest.java` (Surefire: `OptimizePackSizeTest#testPackSizeCostAnalysis`) | Figure 4 panels: `create_chunk_vary_3_plots(...)`. Same run as Figure 3. |
| Figure 11: Improvement of compression ratio of BP-Prune-RMQ and Sprintz-Prune-RMQ | `fig11_improve_compare_ratio.py`, `fig_combine_results.py`; `OptimizePackSizeImpoveCompressionRatio.java` | `BP()`, `BPPruneRMQ()`, `SprintzPruneRMQ()`, `Sprintz()` in `src/OptimizePackSizeImpoveCompressionRatio.java` (Surefire: `OptimizePackSizeImpoveCompressionRatio#BP+BPPruneRMQ+SprintzPruneRMQ+Sprintz`) | Prune-RMQ = Prune (`OptimizePackSizeallV6Plus`) + RMQ. CSV dirs: `results/output_BP_only_Prune_Plus_RMQ_all_no8`, `results/output_Sprintz_only_Prune_Plus_RMQ_all_no8`. Run `fig_combine_results.py` first, then `fig11_improve_compare_ratio.py`. |
| Figure 12: Performance under various fixed pack size 𝑠 | `fig12_vary_pack_size.py`; `OptimizePackSizeVaryPackSize.java` | `BPPruneRMQ()`, `BPPrune()`, `BPAll()`, `SprintzPruneRMQ()`, `SprintzAll()`, `SprintzPrune()`, `BP()`, `Sprintz()` in `src/OptimizePackSizeVaryPackSize.java` (Surefire: `OptimizePackSizeVaryPackSize#BPPruneRMQ+BPPrune+BPAll+SprintzPruneRMQ+SprintzAll+SprintzPrune+BP+Sprintz`) | Edit hard-coded `data_dirs` paths in the script if your CSV output directories differ. |
| Figure 13: Performance under various page sizes 𝑛 | `fig13_vary_page_size.py`; `OptimizePackSizeVaryPageSize.java` | `BP()`, `Sprintz()`, `BPAll()`, `SprintzAll()`, `BPonlyPrune()`, `SprintzonlyPrune()`, `BPPruneRMQ()`, `SprintzPruneRMQ()`, `VaryPageSizeOptimizePackSizeFiltersPlus()`, `VaryPageSizeOptimizePackSizeFiltersPlusSprintz()` in `src/OptimizePackSizeVaryPageSize.java` (Surefire: `OptimizePackSizeVaryPageSize#BP+Sprintz+BPAll+SprintzAll+BPonlyPrune+SprintzonlyPrune+BPPruneRMQ+SprintzPruneRMQ+VaryPageSizeOptimizePackSizeFiltersPlus+VaryPageSizeOptimizePackSizeFiltersPlusSprintz`) | Edit hard-coded `data_dirs` paths in the script if your CSV output directories differ. |
| Figure 15: Compression performance impact after deployment in the real system | `fig15_system.py`; `sprintz_packsize8_vs_optimal_benchmark_test.cc` | `SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder` in **apache/tsfile** `cpp/test/encoding/` | Integer datasets from `integer_datasets.py` → `TestData/` via `scripts/prepare_integer_testdata.py`. CSV: parent `output_tsfile_packsize_comparison_cpp/tsfile_comparison_cpp.csv`. Use `./scripts/run_fig15_system_integer.sh` (see **§4.1**). |
| Figure 14: Pruning rate of candidate pack sizes on various datasets | `fig14_fileter_p_prune_plus.py`; `OptimizePackSizeFilter.java` | `BPPruneRMQ()` and `SprintzPruneRMQ()` in `src/OptimizePackSizeFilter.java` (Surefire: `OptimizePackSizeFilter#BPPruneRMQ+SprintzPruneRMQ`) | Default inputs: `output_BP_filters_plus/` and `output_Sprintz_filters_plus/`. |

### 4.1 Figure 15: integer datasets + TsFile C++ system benchmark

Figure 15 compares **Sprintz** (`PackSize8`) vs **Sprintz-Prune-RMQ** (`OptimalPackSize`) on the **integer** benchmarks listed in `integer_datasets.py` (same set as Fig. 11–14). The plotting script `fig15_system.py` (and parent `../fig_system.py`) reads the C++ CSV and averages over those datasets.

**Prerequisites**

- **Python 3.9+** with `requirements.txt` installed (venv recommended).
- **apache/tsfile** checkout with C++ tests built (`cmake --build build --target TsFile_Test`). The helper script searches, in order: `$TSFILE_CPP_DIR`, `../tsfile/cpp` (sibling of **encoding-pack-size**), `{basedir}/tsfile/cpp`, and `~/Documents/GitHub/tsfile/cpp`.
- Raw integer CSVs under `../integer_dataset/` (see `integer_datasets.py`).

**One command (recommended)**

```bash
cd {basedir}
chmod +x scripts/run_fig15_system_integer.sh   # once
./scripts/run_fig15_system_integer.sh
```

This script:

1. Runs `scripts/prepare_integer_testdata.py` → `TestData/`.
2. Sets `TSFILE_BENCHMARK_DATA_DIR={basedir}/TestData` and runs `SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder`.
3. Writes `../output_tsfile_packsize_comparison_cpp/tsfile_comparison_cpp.csv`.
4. Runs `python3 fig15_system.py` → `../figure_for_paper/system_compare.png` (and `.eps`).

**Manual steps (equivalent)**

```bash
cd {basedir}
python3 scripts/prepare_integer_testdata.py

export TSFILE_BENCHMARK_DATA_DIR="{basedir}/TestData"
export TSFILE_BENCHMARK_WARMUP=0
export TSFILE_BENCHMARK_MEASURE_REPEATS=5

cd /path/to/tsfile/cpp
cmake --build build --target TsFile_Test -j$(sysctl -n hw.ncpu)   # if needed
./build/test/lib/TsFile_Test \
  --gtest_filter=SprintzPackSize8VsOptimal.CsvBenchmarkFairOrder

cd {basedir}
python3 fig15_system.py
```

**macOS build troubleshooting:** If `cmake --build` fails with `'cstddef' file not found` or `'sys/types.h' file not found`, delete `tsfile/cpp/build` and reconfigure with the macOS SDK and `/usr/bin/clang++` (the helper script does this automatically). If ANTLR4 download from GitHub times out, use `-DENABLE_ANTLR4=OFF` — sufficient for the Fig. 15 Sprintz benchmark.

---

## 5. Floating-point experiments: code path and lossless conversion

Floating-point handling on this path is intended to be **lossless** where scaling is applied. The unit test below checks that the decimal scaling logic round-trips without loss under its documented assumptions:

```bash
cd {basedir} && mvn test '-Dtest=FloatToInteger#FloatToIntLosslessTest'
```

(`{basedir}` is the **`pack-size-for-icde27`** directory, as in **§3 Quick path**.)
