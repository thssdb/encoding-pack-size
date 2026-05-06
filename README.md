# encoding-pack-size

Artifacts and Python tooling for the **pack-size optimization** experiments reported in the revised manuscript (Submission #2888). This repository is intended to make **figure reproduction** and **inspection of tabulated metrics** straightforward for reviewers.

---

## 1. Repository snapshot (branch, commit, scope)

| Item | Value |
|------|--------|
| **Default branch** | `main` |
| **Representative commit** | `c2b8ed1841e02a767d7625bf9573032e1abf4dfe` (update this line after each release; run `git rev-parse HEAD`) |

**What this tree contains**

- Python scripts that read experiment outputs (CSV / Excel) and generate paper figures.
- A subset of **public float / numeric CSV datasets** under `TestData/` used in the evaluation.

**What is not vendored here**

- The **full encoder / decoder implementation**, including the **dynamic packing** logic (per-pack-size evaluation, pruning, RMQ-assisted search, and integration with bit-packing and Sprintz), lives in the **compression benchmark harness** that produced the result files referenced by the scripts (see §2). If you are verifying end-to-end behavior, use that harness together with this repository.

---

## 2. Core method and dynamic packing (implementation location)

Reviewers asked for the **dynamic packing** component used in the reported results.

- **In this repository:** analysis and plotting only; no standalone C/C++/Java/Rust compressor sources are checked in.
- **Implementation of the proposed method (including dynamic packing):** maintained in the **same experimental codebase** used to generate the per-dataset logs and spreadsheets consumed below (e.g. rows such as `Sprintz (Prune-RMQ)`, `BP (Prune-RMQ)`, and related variants in `camel_ratio*.xlsx`). Script comments point to that workflow (for example, references to a Java-side evaluator in `fig_combine_results.py`).

**Action for authors:** Replace this paragraph with a **public URL**, **exact branch name**, and **root path** of the benchmark repository once it is linked for the camera-ready / rebuttal artifact, so reviewers can open the encoder and the dynamic pack-size search in one place.

---

## 3. Reproducing main figures and tables (to the extent possible from this tree)

### 3.1 Environment

Use Python 3.9+ (tested with common scientific stacks). Install dependencies:

```bash
cd /path/to/encoding-pack-size
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas matplotlib numpy scipy openpyxl
```

### 3.2 Required inputs (not all are committed)

Several scripts expect **precomputed** experiment outputs next to the repo or under paths you configure:

- `fig2_draw_run_lengths.py` reads `data/features_and_best_p.csv` and `data/features_and_best_p_sprintz.csv` (directory `data/` relative to the repository root). Place these files there after running the feature / optimal-pack-size export from the benchmark, or adjust `_DATA` in the script.
- `fig_combine_results.py`, `fig_compare_ratio.py`, `fig_vary_page_size.py`, and others reference directories such as `./output_BP`, `./output_sprintz`, `./compare_camel/`, etc., and in some places **absolute paths** left from the authors’ machines. **Search and replace** those paths with your local clone before running.

### 3.3 Commands (examples)

After inputs are in place and paths are fixed:

```bash
# Figure 2: optimal pack-size distributions (see §4)
python fig2_draw_run_lengths.py

# Other figure / table helpers (run when their inputs exist)
python fig_compare_ratio.py
python fig_combine_results.py
# …and similarly for fig_cost_vary_pack_size.py, fig_vary_pack_size.py,
# fig_vary_page_size.py, fig_pruning_vary_page_size.py, fig_fileter_p_prune_plus.py
```

Outputs are written under `./figure_for_paper/` (and similar folders) as PNG/EPS where applicable.

---

## 4. Figure ↔ script mapping (paper cross-reference)

|    Manuscript item    | Script | Test Function of Results | Note|
|-----------------------------|--------|--------|--------|
| Figure 2: Distribution of optimal pack sizes for real world datasets (Table 2) compressed by BP and Sprintz | `fig2_draw_run_lengths.py` | `FeatureTest()` and `FeatureAfterSprintzTest()` of `src/AllNo8PacksizeOptimal.java` | Builds combined histograms from `data/features_and_best_p.csv` (BP) and `data/features_and_best_p_sprintz.csv` (Sprintz). |
| Figure 3: Total storage cost of 1024 values of dataset PM10
(Table 2) under various pack sizes 𝑠. It consists of two parts,
the bit width cost and the value cost. | `fig_of_cost_values_bitwidth_in_chunk(csv_dir, chunk_output, chunk_size=1024)` of `fig3_cost_vary_pack_size.py` | `testPackSizeCostAnalysis()` of  `src/AllNo8PacksizeOptimal.java` | |
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

- **Datasets (floating-point columns as stored in CSV):** under `TestData/` (e.g. `City-temp.csv`, `Wind-Speed.csv`, …). These files are the **inputs** to the benchmark runs that produced the compression logs consumed by the figure scripts.
- **Where floating-point values become integers for encoding:** that conversion (scaling, fixed-point representation, ZigZag, or similar) is performed **inside the compression benchmark**, not in the Python files in this repository. This README therefore **cannot** assert losslessness from this tree alone.

**Clarification for reviewers (authors must align with the actual encoder):**

- State in the **benchmark repository README** whether the FP→integer mapping used in the reported experiments is **bit-exact reversible** (lossless for the values present in each column) or **lossy** (e.g. rounding to a fixed grid), and cite the **exact class / function** that performs the mapping.
- If the pipeline uses **textual CSV parsing followed by decimal parsing into a fixed-width integer representation with no rounding beyond representable integers**, say so explicitly; otherwise describe the rounding rule.
