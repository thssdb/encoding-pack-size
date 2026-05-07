# On Optimizing Pack Size for Bit-Packing

Artifacts and Python tooling for the **pack-size optimization** experiments reported in the paper. 
This repository is intended to make **figure reproduction** and **inspection of tabulated metrics** straightforward.

**Fastest path:** **§3 → Quick path**, then **§4 → Quick lookup of script** (copy-paste **Commands** there; use the table below for tests and notes).

---

## 1. Core method and dynamic packing (implementation location)

- **In this repository:** analysis and plotting, plus **standalone JUnit benchmarks** under **`src/`** (package **`encoding.packsize`**, runnable with **`mvn test`** at the repo root; see **§3.2**).
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

Do steps **in this order**; the only figure-specific part is **§4**.

1. **One-time setup** — **§3.1** (`./scripts/bootstrap_env.sh` or manual steps: Python venv, **`requirements.txt`**; **tsfile** clone is optional unless you need the upstream Java tree).
2. **Find your figure** — open **§4 → Quick lookup of script**, use the **Commands** column for copy-paste steps; open the **detailed table** below for Java test names and notes.
3. **Generate inputs (Java)** — run the numbered **`mvn test …`** lines in **Quick lookup of script** from **`{basedir}`** (this repo root; see **§3.2**). Skip `mvn` if that figure’s commands are Python-only (e.g. Figures 19–20).
4. **Plot (Python)** — `cd` into **encoding-pack-size**, activate the venv, run the **`python3 …`** lines from the same row. Figures usually write to `./figure_for_paper/` (see **§3.3**).

`{basedir}` is the root of this **encoding-pack-size** checkout: run both **`mvn test …`** and **`python3 fig*.py`** from `cd {basedir}`. Test sources live under **`src/`** in this repo (package **`encoding.packsize`**); you do not need a **tsfile** checkout for these JUnit runs.

### 3.1 Environment

Use **Python 3.9 or newer**; the commands below call the interpreter as **`python3`** (on some Windows installs, use **`py -3`** in the same places). This repo pins **Python 3.11** in **`mise.toml`** / **`.python-version`** for reproducibility; older 3.9+ interpreters still work if you create the venv manually.

**Declared versions (config files)**

| Role | File | Purpose |
|------|------|---------|
| Python packages | **`requirements.txt`** | `pip install -r requirements.txt` inside the venv |
| Python + JDK (optional) | **`mise.toml`** | If you use [mise](https://mise.jdx.dev/), run **`mise install`** in this repo, then open a new shell |
| Python for pyenv | **`.python-version`** | `pyenv` picks 3.11 when you `cd` here |

**Clone Repository**

```bash
git clone https://github.com/thssdb/encoding-pack-size.git
cd encoding-pack-size
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

After this, your `{basedir}` is typically the `encoding-pack-size` directory (it may also contain a `tsfile/` subfolder if you ran **§3.1** clone steps). If you clone **tsfile** as a sibling of `encoding-pack-size`, keep **`mvn`** / **`python3`** paths consistent with where you keep each checkout.

### 3.2 Required inputs (Java / Maven)

Benchmark CSVs are produced by **JUnit** in this repository. From **`{basedir}`** (the **encoding-pack-size** root), with **JDK 11+** and **Maven 3.6+**:

```bash
cd {basedir}/tsfile/java/tsfile 
mvn install -pl org.apache.tsfile:tsfile-java,org.apache.tsfile:common -DskipTests
mvn test -pl org.apache.tsfile:tsfile '-Dtest=OptimizePackSize#DynamicPackingInTsFile'
```

Use the exact **`mvn test '-Dtest=…'`** selectors in **§4 → Quick lookup of script**. Join methods on the same class with `+`; separate classes with `,` (see [Maven Surefire](https://maven.apache.org/surefire/maven-surefire-plugin/examples/single-test.html)). **`FloatToIntLosslessTest`** lives on **`FloatToInteger`**: `mvn test '-Dtest=FloatToInteger#FloatToIntLosslessTest'` (the same method also exists on **`OptimizePackSize`** / **`OptimizePackSizeTest`** if you prefer those classes).

The **tsfile** branch in **§2** remains the canonical home of the same sources for Apache TsFile integration; you only need that checkout if you are hacking or validating inside the full **tsfile** Maven reactor.

### 3.3 Figures (plotting)

From the **encoding-pack-size** root, with the venv activated:

```bash
cd {basedir}
source .venv/bin/activate   # if not already active
python3 <script-from-section-4>.py
```

Use the **`python3 …`** step(s) in **§4 → Quick lookup of script** for your figure. Path edits and data prerequisites are in the **Note** column of the detailed table. Outputs are usually under `./figure_for_paper/` as PNG/EPS.

---

## 4. Figure <-> script and test functions mapping

### Quick lookup of script

| Quick lookup of script | Figure | Script(s) | Commands (run in order) |
|------------------------|--------|-----------|-------------------------|
| `fig2_draw_run_lengths.py` | 2 | `fig2_draw_run_lengths.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#FeatureTest+FeatureAfterSprintzTest'`<br>2. `cd {basedir} && python3 fig2_draw_run_lengths.py` |
| `fig3_cost_vary_pack_size.py` | 3–4 | `fig3_cost_vary_pack_size.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#testPackSizeCostAnalysis'`<br>2. `cd {basedir} && python3 fig3_cost_vary_pack_size.py` (edit `csv_dir` / `output_dir` in `if __name__ == '__main__'` if needed; one run produces both Fig. 3 and Fig. 4 outputs.) |
| `fig11_data_characters_predict.py` | 11 | `fig11_data_characters_predict.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#OptimizePackSizePruneRMQFeatureOutputTest'`<br>2. `cd {basedir} && python3 fig11_data_characters_predict.py` |
| `fig12_improve_compare_ratio.py` | 12 | `fig12_improve_compare_ratio.py`, then `fig_combine_results.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#BPTest+OptimizePackSizePruneRMQTest+OptimizePackSizePruneRMQSprintzTest+SprintzTest'`<br>2. `cd {basedir} && python3 fig12_improve_compare_ratio.py`<br>3. `cd {basedir} && python3 fig_combine_results.py` |
| `fig13_alp_cuszp.py` | 13 | `fig13_alp_cuszp.py` | 1. `cd {basedir} && mvn test '-Dtest=ALPOptimalPackSize#ALP+ALPPruneRMQ,CuSZp2OptimalPackSize#CuSZp2+CuSZp2PruneRMQ'`<br>2. `cd {basedir} && python3 fig13_alp_cuszp.py` |
| `fig14_compare_baseline.py` | 14 | `fig14_compare_baseline.py` | 1. `cd {basedir} && mvn test '-Dtest=Baseline#BP+Simple8b+BPPruneRMQ,HBPIndexLongTest#BitWeaving'`<br>2. In **SElfStar**: `mvn test -Dtest=TestCompressorPacksize#testAllCompressor` (see Note below)<br>3. `cd {basedir} && python3 fig14_compare_baseline.py` |
| `fig15_vary_pack_size.py` | 15 | `fig15_vary_pack_size.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeVaryPackSize#BPPruneRMQ+BPPrune+BPAll+SprintzPruneRMQ+SprintzAll+SprintzPrune+BP+Sprintz'`<br>2. `cd {basedir} && python3 fig15_vary_pack_size.py` |
| `fig16_vary_page_size.py` | 16 | `fig16_vary_page_size.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeVaryPageSize#BP+Sprintz+BPAll+SprintzAll+BPonlyPrune+SprintzonlyPrune+BPPruneRMQ+SprintzPruneRMQ+OptimizePackSizeFiltersPlus+OptimizePackSizeFiltersPlusSprintz'`<br>2. `cd {basedir} && python3 fig16_vary_page_size.py` |
| `fig17_vary_pack_size_sort.py` | 17 | `fig17_vary_pack_size_sort.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeSort#BPAll+BPAllSort+BPPruneRMQ+BPPruneRMQSort'`<br>2. `cd {basedir} && python3 fig17_vary_pack_size_sort.py` |
| `fig18_vary_pack_size_candidate_limited.py` | 18 | `fig18_vary_pack_size_candidate_limited.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#OptimizePackSizePruneRMQLimitedTest+OptimizePackSizePruneRMQTest'`<br>2. `cd {basedir} && python3 fig18_vary_pack_size_candidate_limited.py` |
| `fig19_simd.py` | 19 | `fig19_simd.py` | 1. `cd {basedir} && python3 fig19_simd.py` |
| `fig20_system.py` | 20 | `fig20_system.py` | 1. `cd {basedir} && python3 fig20_system.py` |
| `fig21_fileter_p_prune_plus.py` | 21 | `fig21_fileter_p_prune_plus.py` | 1. `cd {basedir} && mvn test '-Dtest=OptimizePackSizeTest#OptimizePackSizeFiltersPlusTest+OptimizePackSizeFiltersSprintzPlusTest'`<br>2. `cd {basedir} && python3 fig21_fileter_p_prune_plus.py` |

**Surefire:** run from **`{basedir}`** with a quoted **`-Dtest=Class#method…`** selector, as in **§3.2** and the **Commands (run in order)** column of **Quick lookup of script**. Set `{basedir}` as in **§3 Quick path** and **§3.1**.

### Detailed mapping (tests and notes)

| Manuscript item | Script | Test function(s) of results | Note |
|-----------------|--------|------------------------------|------|
| Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz | `fig2_draw_run_lengths.py` | `FeatureTest()` and `FeatureAfterSprintzTest()` of `src/OptimizePackSizeTest.java` | Builds combined histograms from `data/features_and_best_p.csv` (BP) and `data/features_and_best_p_sprintz.csv` (Sprintz). |
| Figure 3: Total storage cost of 1024 values of dataset PM10 (Table 2) under various pack sizes 𝑠. It consists of two parts, the bit width cost and the value cost. | `fig_of_cost_values_bitwidth_in_chunk(...)` in `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of `src/OptimizePackSizeTest.java` | |
| Figure 4: Example on unimodality of the cost function 𝐶(𝑠) on a subset of pack sizes 𝑠 = 3*2^𝛽 , 𝛽 = 0, 1, 2, . . . , according to Proposition 1 | `create_chunk_vary_3_plots(...)` in `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of `src/OptimizePackSizeTest.java` | Same run as Figure 3; see **Quick lookup of script** row **3–4**. |
| Figure 11: Data characters predict the impact of optimal pack sizes | `fig11_data_characters_predict.py` | `OptimizePackSizePruneRMQFeatureOutputTest()` of `src/OptimizePackSizeTest.java` | |
| Figure 12: Improvement of compression ratio of BP-Prune-RMQ and Sprintz-Prune-RMQ | `fig12_improve_compare_ratio.py` and `fig_combine_results.py` | `BPTest()`, `OptimizePackSizePruneRMQTest`, `OptimizePackSizePruneRMQSprintzTest()`, `SprintzTest()` of `src/OptimizePackSizeTest.java` | |
| Figure 13: The compression performance of ALP and CuSZp2 with (without) optimal pack size | `fig13_alp_cuszp.py` | `test0()` and `test1_optimalPackV5()` of `src/ALPTest.java`, `cuSZpCpu1DTest()` and `cuSZpCpu1DOptimalV5Test()` of `src/CuSZpCpuTest.java` | |
| Figure 14: Comparison with Other Algorithms | `fig14_compare_baseline.py` | `BP()`, `Simple8b()` and `BPPruneRMQ()` of `src/Baseline.java`, `BitWeaving()` of `src/HBPIndexLongTest.java`, `testAllCompressor()` of `TestCompressorPacksize.java` | put `TestCompressorPacksize.java` into `SElfStar/src/test/java/` of https://github.com/Spatio-Temporal-Lab/SElfStar |
| Figure 15: Performance under various fixed pack size 𝑠 | `fig15_vary_pack_size.py` | `OptimizePackSizePruneRMQ()`, `OptimizePackSizePrunePlus()`, `OptimizePackSizeRMQSprintz()`, `OptimizePackSizeN2Sprintz()`, `OptimizePackSizePrunePlusSprintz()`, `VaryPackSize()`, `VaryPackSizeSprintz()` in `src/OptimizePackSizeVaryPackSize.java` (paper row “AllSprintz” → **`OptimizePackSizeN2Sprintz`** in code) | |
| Figure 16: Performance under various page sizes 𝑛 | `fig16_vary_page_size.py` | `BP()`, `Sprintz()`, …, `OptimizePackSizeFiltersPlus()`, `OptimizePackSizeFiltersPlusSprintz()` in `src/OptimizePackSizeVaryPageSize.java` (same suite as **Quick lookup** row **16**) | |
| Figure 17: Impact of sorting non-time-series data on bit-packing with optimal pack size | `fig17_vary_pack_size_sort.py` | `BPAll()`, `BPAllSort()`, `BPPruneRMQ()`, `BPPruneRMQSort()` in `src/OptimizePackSizeSort.java`; `VaryPackSizeTest()`, `VaryPackSizeSortTest()`, `VaryPackSizeSprintzTest()`, `VaryPackSizeSprintzSortTest()` in `src/OptimizePackSizeTest.java` | |
| Figure 18: Compression performance of bit-packing with pre-defined pack sizes and all pack sizes | `fig18_vary_pack_size_candidate_limited.py` | `OptimizePackSizePruneRMQLimitedTest()` and `OptimizePackSizePruneRMQTest()` of `src/OptimizePackSizeTest.java` | |
| Figure 19: Compression performance of bit-packing optimal pack size with SIMDComp, Fastlane and SIMT | `fig19_simd.py` | — | Consumes CSVs under this repo (e.g. `output_simd/`); generate those with your SIMD experiment pipeline if you are not using bundled example data. |
| Figure 20: Compression performance impact after deployment in the real system | `fig20_system.py` | — | Expects `output_tsfile_packsize_comparison_cpp/tsfile_comparison_cpp.csv` (paths in the script may need editing). |
| Figure 21: Pruning rate of candidate pack sizes on various datasets | `fig21_fileter_p_prune_plus.py` | `OptimizePackSizeFiltersPlusTest()` and `OptimizePackSizeFiltersSprintzPlusTest()` of `src/OptimizePackSizeTest.java` | |

---

## 5. Floating-point experiments: code path and lossless conversion

Floating-point handling on this path is intended to be **lossless** where scaling is applied. The unit test below checks that the decimal scaling logic round-trips without loss under its documented assumptions:

```bash
cd {basedir} && mvn test '-Dtest=FloatToInteger#FloatToIntLosslessTest'
```

(`{basedir}` is the same root as in **§3 Quick path**—typically the **encoding-pack-size** checkout.)