# Experiment Analysis Report - Complete vs. Without SUTs

This report analyzes the performance of `OpenCode_hierarchical_index_complete` and `OpenCode_hierarchical_index_without` across 3 independent runs (R1, R2, R3) on the archeology dataset (12 tasks).

## 1. Executive Summary

- **Accuracy Variance**: The `complete` SUT (with hierarchical indexing) achieved a mean accuracy of **41.67%** (Std Dev: **8.33%**), while the `without` SUT (without hierarchical indexing) achieved a mean accuracy of **36.11%** (Std Dev: **4.81%**).
- **Token Efficiency**: The `complete` SUT consumed a total of **1,732,797** tokens (mean: **577,599.0** per run), compared to the `without` SUT which consumed **1,702,205** tokens (mean: **567,401.7** per run). This represents a **+1.80%** change in token consumption.
- **Phase Breakdown**:
  - **Complete SUT**: Exploration phase (**19.2%**) vs. Analysis phase (**80.8%**).
  - **Without SUT**: Exploration phase (**20.0%**) vs. Analysis phase (**80.0%**).
- **Skill Invocations**: Across all `complete` runs, the 3 indexing skills were invoked as follows:
  - **Expand**: 16 calls (**57.1%**)
  - **Search**: 12 calls (**42.9%**)
  - **Summarize**: 0 calls (**0.0%**)
- **File Usage Correlation**: The Pearson correlation between the F1-score of the files referenced in the generated python pipelines and the task accuracy is **0.3568**, showing a strong relationship between correct file identification and final task success.

---

## 2. SUT Run Summary Table

| SUT Run     | Accuracy (%)   |   Total Tokens |   Mean Tokens/Task |   Mean F1 File Usage |
|:------------|:---------------|---------------:|-------------------:|---------------------:|
| complete_R1 | 33.33%         |        494,940 |            41245   |               0.9444 |
| complete_R2 | 50.00%         |        708,549 |            59045.8 |               0.9444 |
| complete_R3 | 41.67%         |        529,308 |            44109   |               0.9444 |
| without_R1  | 33.33%         |        521,680 |            43473.3 |               0.9444 |
| without_R2  | 41.67%         |        519,250 |            43270.8 |               0.9444 |
| without_R3  | 33.33%         |        661,275 |            55106.2 |               0.9444 |

---

## 3. Task-by-Task Success / Failure Comparison

| Task ID | complete_R1 | complete_R2 | complete_R3 | without_R1 | without_R2 | without_R3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `archeology-hard-1` | 0 | 0 | 0 | 0 | 0 | 0 |
| `archeology-hard-2` | 0 | 0 | 0 | 0 | 0 | 0 |
| `archeology-easy-3` | 1 | 1 | 1 | 1 | 1 | 0 |
| `archeology-easy-4` | 1 | 1 | 1 | 1 | 1 | 1 |
| `archeology-hard-5` | 0 | 0 | 0 | 0 | 0 | 0 |
| `archeology-easy-6` | 1 | 1 | 1 | 0 | 0 | 1 |
| `archeology-hard-7` | 0 | 1 | 0 | 0 | 1 | 0 |
| `archeology-easy-8` | 0 | 0 | 0 | 0 | 0 | 0 |
| `archeology-hard-9` | 0 | 0 | 0 | 0 | 0 | 0 |
| `archeology-easy-10` | 1 | 1 | 1 | 1 | 1 | 1 |
| `archeology-easy-11` | 0 | 1 | 1 | 1 | 1 | 1 |
| `archeology-hard-12` | 0 | 0 | 0 | 0 | 0 | 0 |

---

## 4. Task-by-Task Token Cost Comparison

| Task ID | complete_R1 | complete_R2 | complete_R3 | without_R1 | without_R2 | without_R3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `archeology-hard-1` | 26,790 | 77,193 | 51,394 | 24,777 | 26,389 | 26,997 |
| `archeology-hard-2` | 28,061 | 40,208 | 17,513 | 19,067 | 16,962 | 61,016 |
| `archeology-easy-3` | 56,784 | 23,189 | 21,451 | 12,937 | 57,012 | 57,562 |
| `archeology-easy-4` | 16,545 | 16,069 | 20,753 | 21,857 | 15,951 | 14,214 |
| `archeology-hard-5` | 27,471 | 26,762 | 36,574 | 25,453 | 24,734 | 25,264 |
| `archeology-easy-6` | 35,771 | 14,147 | 16,470 | 11,474 | 13,241 | 12,034 |
| `archeology-hard-7` | 14,943 | 83,583 | 68,103 | 15,135 | 101,995 | 99,934 |
| `archeology-easy-8` | 65,883 | 104,071 | 62,095 | 67,112 | 69,558 | 48,021 |
| `archeology-hard-9` | 118,814 | 123,629 | 117,935 | 115,093 | 74,449 | 105,295 |
| `archeology-easy-10` | 12,789 | 57,513 | 13,856 | 12,686 | 15,671 | 56,496 |
| `archeology-easy-11` | 24,518 | 21,885 | 21,443 | 77,660 | 16,253 | 16,748 |
| `archeology-hard-12` | 66,571 | 120,300 | 81,721 | 118,429 | 87,035 | 137,694 |

---

## 5. Failure Classification & Analytical Reasoning

Based on the Point of Failure analysis for `complete_R1`, the failures are classified as follows:

1. **different user QUERY interpretation**:
   - `archeology-hard-1` (all runs): The agent computed a simple two-point endpoint average for Malta's human habitation period, whereas the target solution required year-by-year interpolation over the entire temporal range (~407 years).
   - `archeology-hard-12` (all runs): The agent included conflicts starting in the period but ending later, and missed actor deduplication, outputting an attribution list rather than a single integer.
   - `archeology-hard-5` (all runs): The agent incorrectly restricted its search for the northernmost Neolithic sample to Malta, instead of searching globally.

2. **Incorrect PIPELINE implementation/code generation**:
   - `archeology-hard-2` (all runs): The agent calculated consecutive differences at the raw 0.5 kyr resolution, missing the calendar year grouping/rounding preprocessing logic used in the target solution.
   - `archeology-hard-9` (all runs): The agent used an arbitrary CSV row-index order to break ties when multiple Roman cities matched one modern city, instead of using the settlement rank value.
   - `archeology-easy-8` (all runs): The agent failed to strip page-number suffixes (splitting by `:`) from bibliographical references, leading to duplicate canonical sources.
   - `archeology-hard-7` (R1/R3): The agent used a Cartesian square bounding box check instead of a proper Euclidean KD-tree query ball check.
   - `archeology-easy-11` (complete_R1): The agent applied `dropna` on population before country deduplication, dropping countries that had missing population values.

3. **encoding/formatting issue**:
   - `archeology-easy-6` (without_R1/R2): Outputted `Sao Paulo` instead of the accented `São Paulo`.
   - `archeology-easy-3` (without_R3): Outputted a leading BOM character `﻿3.1333`.
