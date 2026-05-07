# On Optimizing Pack Size for Bit-Packing

Artifacts and Python tooling for the **pack-size optimization** experiments reported in the paper. 
This repository is intended to make **figure reproduction** and **inspection of tabulated metrics** straightforward.

**Fastest path:** **§3 → Quick path**, then **§4 → Quick lookup** (copy-paste **Commands** there; use the table below for tests and notes).

---

## 1. Core method and dynamic packing (implementation location)

- **In this repository:** analysis and plotting only.
- **Implementation of the proposed method (including dynamic packing):** maintained in the **same experimental codebase** used to generate the csv consumed below (e.g. rows such as `Sprintz (Prune-RMQ)`, `BP (Prune-RMQ)`). 
Script comments point to that workflow (for example, references to a Java-side evaluator in `fig_combine_results.py`).

---

## 2. Repository snapshot (Repository link, branch, and commit)

### The repository repo1 of scripts

Repository link: https://github.com/thssdb/encoding-pack-size

Branch: main

Commit hash: 8235cf0a0f926b09fe7cffc9997243aedd71ce38

### The repository repo2 of the core implementation

Repository link: https://github.com/apache/tsfile/tree/research/encoding-pack-size/

Branch: research/encoding-pack-size

Commit hash: 0157b8a4dc7ada165106d231e9170d629ad80853


---

## 3. Commands to run

### Quick path (reproduce one figure)

Do steps **in this order**; the only figure-specific part is **§4**.

1. **One-time setup** — **§3.1** (Python venv, clone this repo and **tsfile**).
2. **Find your figure** — open **§4 → Quick lookup**, use the **Commands** column for copy-paste steps; open the **detailed table** below for Java test names and notes.
3. **Generate inputs (Java)** — run the numbered **`mvn test …`** lines in **Quick lookup** (from the **tsfile** tree; see **§3.2** for a generic build example). Skip `mvn` if that figure’s commands are Python-only (e.g. Figures 19–20).
4. **Plot (Python)** — `cd` into **encoding-pack-size**, activate the venv, run the **`python …`** lines from the same row. Figures usually write to `./figure_for_paper/` (see **§3.3**).

`{basedir}` is the root of this **encoding-pack-size** checkout: run `python fig*.py` from `cd {basedir}`. After **§3.1**, that directory usually also contains `tsfile/`; if **tsfile** lives elsewhere, set `{basedir}` to the folder that contains your `tsfile` tree and run the Python steps from your actual script checkout.

### 3.1 Environment

Use Python 3.9+. Install dependencies and clone repositories:

```bash
git clone https://github.com/thssdb/encoding-pack-size.git
cd encoding-pack-size
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas matplotlib numpy scipy openpyxl
git clone -b research/encoding-pack-size --single-branch https://github.com/apache/tsfile/
```

After this, your `{basedir}` is typically the `encoding-pack-size` directory (it now contains a `tsfile/` subfolder). If you clone **tsfile** as a sibling of `encoding-pack-size`, `{basedir}` is their common parent instead.

### 3.2 Required inputs (Java / Maven)

Experimental artifacts (CSV logs, etc.) come from the **tsfile** branch in **§2**. Build and run tests from your **tsfile** Java tree (paths may differ slightly from **§4** if you use `-pl …`; adjust to match your layout). Example:

```bash
cd tsfile/java 
mvn install -pl org.apache.tsfile:tsfile-java,org.apache.tsfile:common -DskipTests
mvn test -pl org.apache.tsfile:tsfile '-Dtest=OptimalPackSize#OptimalPackSizePruneRMQTest'
```

For each figure, use the exact **`mvn test -Dtest=…`** lines in **§4 → Quick lookup**. Join methods on the same class with `+`; separate classes with `,` (see [Maven Surefire](https://maven.apache.org/surefire/maven-surefire-plugin/examples/single-test.html)).

### 3.3 Figures (plotting)

From the **encoding-pack-size** root, with the venv activated:

```bash
cd {basedir}
source .venv/bin/activate   # if not already active
python <script-from-section-4>.py
```

Use the **`python …`** step(s) in **§4 → Quick lookup** for your figure. Path edits and data prerequisites are in the **Note** column of the detailed table. Outputs are usually under `./figure_for_paper/` as PNG/EPS.

---

## 4. Figure <-> script and test functions mapping

### Quick lookup

| Figure | Script(s) | Commands (run in order) |
|--------|-----------|-------------------------|
| 2 | `fig2_draw_run_lengths.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#FeatureTest+FeatureAfterSprintzTest`<br>2. `cd {basedir} && python fig2_draw_run_lengths.py` |
| 3–4 | `fig3_cost_vary_pack_size.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#testPackSizeCostAnalysis`<br>2. `cd {basedir} && python fig3_cost_vary_pack_size.py` (edit `csv_dir` / `output_dir` in `if __name__ == '__main__'` if needed; one run produces both Fig. 3 and Fig. 4 outputs.) |
| 11 | `fig11_data_characters_predict.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#OptimalPackSizePruneRMQFeatureOutputTest`<br>2. `cd {basedir} && python fig11_data_characters_predict.py` |
| 12 | `fig12_improve_compare_ratio.py`, then `fig_combine_results.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#BPTest+OptimalPackSizePruneRMQTest+OptimalPackSizePruneRMQSprintzTest+SprintzTest`<br>2. `cd {basedir} && python fig12_improve_compare_ratio.py`<br>3. `cd {basedir} && python fig_combine_results.py` |
| 13 | `fig13_alp_cuszp_pack8_vs_v5_combined.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=ALPTest#test0+test1_optimalPackV5,CuSZpCpuTest#cuSZpCpu1DTest+cuSZpCpu1DOptimalV5Test`<br>2. `cd {basedir} && python fig13_alp_cuszp_pack8_vs_v5_combined.py` |
| 14 | `fig14_compare_baseline.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#BPTest+Simple8bTest+OptimalPackSizePruneRMQTest,HBPIndexLongTest#test0`<br>2. In **SElfStar**: `mvn test -Dtest=TestCompressorPacksize#testAllCompressor` (see Note below)<br>3. `cd {basedir} && python fig14_compare_baseline.py` |
| 15 | `fig15_vary_pack_size.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#OptimalPackSizePruneRMQTest+OptimalPackSizePrunePlusTest+OptimalPackSizeRMQSprintzTest+OptimalPackSizeN2SprintzTest+OptimalPackSizePrunePlusSprintzTest+VaryPackSizeTest+VaryPackSizeSprintzTest`<br>2. `cd {basedir} && python fig15_vary_pack_size.py` |
| 16 | `fig16_vary_page_size.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#TestVariablePageSizeBP+TestVariablePageSizeSprintz+TestVariablePageSizeBPN2+TestVariablePageSizeSprintzN2+TestVariablePageSizeBPonlyPrune+TestVariablePageSizeSprintzonlyPrune+TestVariablePageSizeBPPruneRMQ+TestVariablePageSizeSprintzPruneRMQ+VaryPageSizeOptimalPackSizeFiltersPlusTest+VaryPageSizeOptimalPackSizeFiltersPlusSprintzTest`<br>2. `cd {basedir} && python fig16_vary_page_size.py` |
| 17 | `fig17_vary_pack_size_sort.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#VaryPackSizeTest+VaryPackSizeSortTest+VaryPackSizeSprintzTest+VaryPackSizeSprintzSortTest`<br>2. `cd {basedir} && python fig17_vary_pack_size_sort.py` |
| 18 | `fig18_vary_pack_size_candidate_limited.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#OptimalPackSizePruneRMQLimitedTest+OptimalPackSizePruneRMQTest`<br>2. `cd {basedir} && python fig18_vary_pack_size_candidate_limited.py` |
| 19 | `fig19_simd.py` | 1. `cd {basedir} && python fig19_simd.py` |
| 20 | `fig20_system.py` | 1. `cd {basedir} && python fig20_system.py` |
| 21 | `fig21_fileter_p_prune_plus.py` | 1. `cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#OptimalPackSizeFiltersPlusTest+OptimalPackSizeFiltersSprintzPlusTest`<br>2. `cd {basedir} && python fig21_fileter_p_prune_plus.py` |

**Surefire:** use `{basedir}/tsfile/java/tsfile` as in the commands above when your layout matches; if you use `tsfile/java` with `mvn -pl …` (**§3.2**), keep the same `-Dtest=…` selectors. Set `{basedir}` as in **§3 Quick path** and **§3.1**.

### Detailed mapping (tests and notes)

| Manuscript item | Script | Test function(s) of results | Note |
|-----------------|--------|------------------------------|------|
| Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz | `fig2_draw_run_lengths.py` | `FeatureTest()` and `FeatureAfterSprintzTest()` of `src/OptimalPackSize.java` | Builds combined histograms from `data/features_and_best_p.csv` (BP) and `data/features_and_best_p_sprintz.csv` (Sprintz). |
| Figure 3: Total storage cost of 1024 values of dataset PM10 (Table 2) under various pack sizes 𝑠. It consists of two parts, the bit width cost and the value cost. | `fig_of_cost_values_bitwidth_in_chunk(...)` in `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of `src/OptimalPackSize.java` | |
| Figure 4: Example on unimodality of the cost function 𝐶(𝑠) on a subset of pack sizes 𝑠 = 3*2^𝛽 , 𝛽 = 0, 1, 2, . . . , according to Proposition 1 | `create_chunk_vary_3_plots(...)` in `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of `src/OptimalPackSize.java` | Same run as Figure 3; see **Quick lookup** row **3–4**. |
| Figure 11: Data characters predict the impact of optimal pack sizes | `fig11_data_characters_predict.py` | `OptimalPackSizePruneRMQFeatureOutputTest()` of `src/OptimalPackSize.java` | |
| Figure 12: Improvement of compression ratio of BP-Prune-RMQ and Sprintz-Prune-RMQ | `fig12_improve_compare_ratio.py` and `fig_combine_results.py` | `BPTest()`, `OptimalPackSizePruneRMQTest`, `OptimalPackSizePruneRMQSprintzTest()`, `SprintzTest()` of `src/OptimalPackSize.java` | |
| Figure 13: The compression performance of ALP and CuSZp2 with (without) optimal pack size | `fig13_alp_cuszp_pack8_vs_v5_combined.py` | `test0()` and `test1_optimalPackV5()` of `src/ALPTest.java`, `cuSZpCpu1DTest()` and `cuSZpCpu1DOptimalV5Test()` of `src/CuSZpCpuTest.java` | |
| Figure 14: Comparison with Other Algorithms | `fig14_compare_baseline.py` | `BPTest()`, `Simple8bTest()` and `OptimalPackSizePruneRMQTest()` of `src/OptimalPackSize.java`, `test0()` of `src/HBPIndexLongTest.java`, `testAllCompressor()` of `TestCompressorPacksize.java` | put `TestCompressorPacksize.java` into `SElfStar/src/test/java/` of https://github.com/Spatio-Temporal-Lab/SElfStar |
| Figure 15: Performance under various fixed pack size 𝑠 | `fig15_vary_pack_size.py` | `OptimalPackSizePruneRMQTest()`, `OptimalPackSizePrunePlusTest()`, `OptimalPackSizeRMQSprintzTest`, `OptimalPackSizeN2SprintzTest()`, `OptimalPackSizePrunePlusSprintzTest`, `VaryPackSizeTest()` and `VaryPackSizeSprintzTest()` of `src/OptimalPackSize.java` | |
| Figure 16: Performance under various page sizes 𝑛 | `fig16_vary_page_size.py` | `TestVariablePageSizeBP()`, `TestVariablePageSizeSprintz()`, `TestVariablePageSizeBPN2()`, `TestVariablePageSizeSprintzN2`, `TestVariablePageSizeBPonlyPrune()`, `TestVariablePageSizeSprintzonlyPrune()`, `TestVariablePageSizeBPPruneRMQ()`, `TestVariablePageSizeSprintzPruneRMQ()`, `VaryPageSizeOptimalPackSizeFiltersPlusTest()`, and `VaryPageSizeOptimalPackSizeFiltersPlusSprintzTest()` of `src/OptimalPackSize.java` | |
| Figure 17: Impact of sorting non-time-series data on bit-packing with optimal pack size | `fig17_vary_pack_size_sort.py` | `VaryPackSizeTest()`, `VaryPackSizeSortTest()`, `VaryPackSizeSprintzTest()`, `VaryPackSizeSprintzSortTest()` of `src/OptimalPackSize.java` | |
| Figure 18: Compression performance of bit-packing with pre-defined pack sizes and all pack sizes | `fig18_vary_pack_size_candidate_limited.py` | `OptimalPackSizePruneRMQLimitedTest()` and `OptimalPackSizePruneRMQTest()` of `src/OptimalPackSize.java` | |
| Figure 19: Compression performance of bit-packing optimal pack size with SIMDComp, Fastlane and SIMT | `fig19_simd.py` | — | Consumes CSVs under this repo (e.g. `output_simd/`); generate those with your SIMD experiment pipeline if you are not using bundled example data. |
| Figure 20: Compression performance impact after deployment in the real system | `fig20_system.py` | — | Expects `output_tsfile_packsize_comparison_cpp/tsfile_comparison_cpp.csv` (paths in the script may need editing). |
| Figure 21: Pruning rate of candidate pack sizes on various datasets | `fig21_fileter_p_prune_plus.py` | `OptimalPackSizeFiltersPlusTest()` and `OptimalPackSizeFiltersSprintzPlusTest()` of `src/OptimalPackSize.java` | |

---

## 5. Floating-point experiments: code path and lossless conversion

Floating-point handling on this path is intended to be **lossless** where scaling is applied. The unit test below checks that the decimal scaling logic round-trips without loss under its documented assumptions:

```bash
cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=OptimalPackSize#FloatToIntLosslessTest
```

(`{basedir}` is the same root as in **§3 Quick path**—typically the **encoding-pack-size** directory that contains `tsfile/`.)