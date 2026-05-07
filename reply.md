# Subject: Repository Update for Submission #2888 – Implementation, Reproducibility, and Floating-Point Clarification

Dear PC Chairs,

In response to the meta-reviewer’s request regarding Submission #2888, we have updated the repository and provide the following details.

The repository repo1 of scripts: https://github.com/thssdb/encoding-pack-size
Branch: main
Commit hash: 8235cf0a0f926b09fe7cffc9997243aedd71ce38

The repository repo2 of the core implementation: https://github.com/apache/tsfile
Branch: research/encoding-pack-size
Commit hash: 0157b8a4dc7ada165106d231e9170d629ad80853

Below we address each of the four requested items.

## 1. Core implementation (including the dynamic packing component)

The core implementation of the proposed method resides in the "src/" directory of repo1 or "java/tsfile/src/test/java/org/apache/tsfile/encoding/AllNo8PacksizeOptimal.java" of repo2. In particular, the dynamic packing component is implemented in AllNo8PacksizeOptimal.java, within the method OptimalPackSizePruneRMQTest().

## 2. Scripts, configuration files, and reproduction instructions

The scripts, configuration files, and step-by-step instructions for reproducing the main reported experiments are provided in Sections 2 and 3 of README in repo1.

## 3. Branch, commit, and run commands in the README

The current version of the README in repo1 explicitly specifies the branch, the commit hash, and the exact commands required to reproduce the experiments.

## 4. Floating-point experiments and lossless conversion

The code for the floating-point experiments is located in "AllNo8PacksizeOptimal.java". The floating-point-to-integer conversion is handled by the variable decimalMax (which captures the maximum number of decimal places) and the method scaleNumbers(List<String> numbers, int decimalMax).
This conversion is lossless, the included test functions can be used to verify that no precision is lost during the transformation, which can be verified by "scaleNumbersMultiplyByTenPowDecimalMaxIsLosslesslyInvertible()" in AllNo8PacksizeOptimal.java. Please refer to Section 5 of README in repo1 for the exact commands to run the experiments.

Please feel free to reach out if any further clarification is needed.

Sincerely,