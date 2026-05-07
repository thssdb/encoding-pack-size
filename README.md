# On Optimizing Pack Size for Bit-Packing

Artifacts and Python tooling for the **pack-size optimization** experiments reported in the paper. 
This repository is intended to make **figure reproduction** and **inspection of tabulated metrics** straightforward.

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

### 3.1 Environment

Use Python 3.9+. Install dependencies:

```bash
cd /path/to/encoding-pack-size
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas matplotlib numpy scipy openpyxl
```

### 3.2 Required inputs

Experimental artifacts (CSV logs, etc.) are produced from the **tsfile** checkout described in §1. Replace `{basedir}` with the root directory of your local **tsfile** clone (the repository that contains `java/tsfile`). Example:

```bash
cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=AllNo8PacksizeOptimal#OptimalPackSizePruneRMQTest
```

Use the Maven targets that match each figure or analysis (see §4 for the full figure ↔ script ↔ test mapping).

### 3.3 Figures (plotting)

After the inputs exist and any script paths are configured, generate figures with the Python scripts—for example:

```bash
# Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz (see §4)
python fig2_draw_run_lengths.py
# ... additional scripts for other figures (see §4)
python fig21_fileter_p_prune_plus.py
```

Outputs are written under `./figure_for_paper/` as PNG/EPS where applicable.

---

## 4. Figure <-> script and test functions mapping

- 

|    Manuscript item    | Script | Test Function of Results | Note|
|-----------------------------|--------|--------|--------|
| Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz | `fig2_draw_run_lengths.py` | `FeatureTest()` and `FeatureAfterSprintzTest()` of `src/AllNo8PacksizeOptimal.java` | Builds combined histograms from `data/features_and_best_p.csv` (BP) and `data/features_and_best_p_sprintz.csv` (Sprintz). |
| Figure 3: Total storage cost of 1024 values of dataset PM10 (Table 2) under various pack sizes 𝑠. It consists of two parts, the bit width cost and the value cost. | `fig_of_cost_values_bitwidth_in_chunk(csv_dir, chunk_output, chunk_size=1024)` of `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of  `src/AllNo8PacksizeOptimal.java` | |
| Figure 4: Example on unimodality of the cost function 𝐶(𝑠) on a subset of pack sizes 𝑠 = 3*2^𝛽 , 𝛽 = 0, 1, 2, . . . , according to Proposition 1 |  `create_chunk_vary_3_plots(csv_dir, chunk_output, chunk_size=1024)` of `fig3_cost_vary_pack_size.py`  | `testPackSizeCostAnalysis()` of  `src/AllNo8PacksizeOptimal.java` | |
| Figure 11: Data characters predict the impact of optimal pack sizes |  `fig11_data_characters_predict.py` | `OptimalPackSizePruneRMQFeatureOutputTest()` of `src/AllNo8PacksizeOptimal.java` | |
| Figure 12: Improvement of compression ratio of BP-Prune-RMQ and Sprintz-Prune-RMQ | `fig12_improve_compare_ratio.py` | `BPTest()`, `OptimalPackSizePruneRMQTest`,`OptimalPackSizePruneRMQSprintzTest()`,`SprintzTest()` of `src/AllNo8PacksizeOptimal.java`  and `fig_combine_results.py`| |
| Figure 13: The compression performance of ALP and CuSZp2 with (without) optimal pack size | `fig13_alp_cuszp_pack8_vs_v5_combined.py` | `test0()` and `test1_optimalPackV5()` of `src/ALPTest.java`, `cuSZpCpu1DTest()` and `cuSZpCpu1DOptimalV5Test()` of `src/CuSZpCpuTest.java` | |
| Figure 14: Comparison with Other Algorithms | `fig14_compare_baseline.py` | `BPTest()`, `Simple8bTest()` and `OptimalPackSizePruneRMQTest()` of `src/AllNo8PacksizeOptimal.java`, `test0()` of `src/HBPIndexLongTest.java`, `testAllCompressor()` of `TestCompressorPacksize.java` | put `TestCompressorPacksize.java` into `SElfStar/src/test/java/` of https://github.com/Spatio-Temporal-Lab/SElfStar |
| Figure 15: Performance under various fixed pack size 𝑠 | `fig15_vary_pack_size.py` | `OptimalPackSizePruneRMQTest()`, `OptimalPackSizePruneRMQTest()`, `OptimalPackSizePrunePlusTest()`,`OptimalPackSizeRMQSprintzTest`, `OptimalPackSizeN2SprintzTest()`, `OptimalPackSizePrunePlusSprintzTest`,  `VaryPackSizeTest()` and `VaryPackSizeSprintzTest()` of `src/AllNo8PacksizeOptimal.java`  | |
| Figure 16: Performance under various page sizes 𝑛 | `fig16_vary_page_size.py` |`TestVariablePageSizeBP()`,`TestVariablePageSizeSprintz()`,`TestVariablePageSizeBPN2()`,`TestVariablePageSizeSprintzN2`,`TestVariablePageSizeBPonlyPrune()`,`TestVariablePageSizeSprintzonlyPrune()`, `TestVariablePageSizeBPPruneRMQ()`, `TestVariablePageSizeSprintzPruneRMQ()`, `VaryPageSizeOptimalPackSizeFiltersPlusTest()`, and `VaryPageSizeOptimalPackSizeFiltersPlusSprintzTest()` of `src/AllNo8PacksizeOptimal.java` |   |
| Figure 17: Impact of sorting non-time-series data on bit-packing with optimal pack size | `fig17_vary_pack_size_sort.py` | `VaryPackSizeTest()`,`VaryPackSizeSortTest()`,`VaryPackSizeSprintzTest()`,`VaryPackSizeSprintzSortTest()` of `src/AllNo8PacksizeOptimal.java` |  |
| Figure 18: Compression performance of bit-packing with pre-defined pack sizes and all pack sizes | `fig18_vary_pack_size_candidate_limited.py` | `OptimalPackSizePruneRMQLimitedTest()` and `OptimalPackSizePruneRMQTest()` of `src/AllNo8PacksizeOptimal.java`  | |
| Figure 19: Compression performance of bit-packing optimal pack size with SIMDComp, Fastlane and SIMT | `fig19_simd.py` |  | |
| Figure 20: Compression performance impact after deployment in the real system | `fig20_system.py` | | |
| Figure 21: Pruning rate of candidate pack sizes on various datasets | `fig21_fileter_p_prune_plus.py` | `OptimalPackSizeFiltersPlusTest()` and `OptimalPackSizeFiltersSprintzPlusTest()` of `src/AllNo8PacksizeOptimal.java` | |

---

## 5. Floating-point experiments: code path and lossless conversion

Floating-point handling on this path is intended to be **lossless** where scaling is applied. The unit test below checks that the decimal scaling logic round-trips without loss under its documented assumptions:

```bash
cd {basedir}/tsfile/java/tsfile && mvn test -Dtest=AllNo8PacksizeOptimal#FloatToIntLosslessTest
```

(`{basedir}` is the root of your local **tsfile** clone, as in §3.2.)