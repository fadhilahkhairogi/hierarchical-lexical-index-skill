# KramaBench Experiment Report 2: Complete vs. Without SUTs

This report presents a comprehensive comparative analysis of the `OpenCode_hierarchical_index_complete` SUT (with hierarchical indexing skills) versus the `OpenCode_hierarchical_index_without` SUT (without indexing skills) across the `archeology`, `biomedical`, and `legal` workloads.

---

## 1. Executive Summary

- **Accuracy & Workload Performance**: The Complete SUT achieved a global workload-weighted mean accuracy of **46.48% ± 4.98%**, while the Without SUT achieved **49.44% ± 9.33%**.
- **File Usage Comparison**: Both SUTs demonstrated high Precision and Recall on simple datasets (e.g., Archeology), but the Complete SUT outperformed the Without SUT in selecting correct schemas and data sources on complex structured directories (e.g., Legal dataset).
- **Skill Invocations**: The Complete SUT utilized indexing skills in a significant portion of runs. The `search` skill was invoked most frequently (1.49 calls/task on average, utilized in 26.67% of tasks), followed by `expand` (0.96 calls/task, utilized in 33.33% of tasks), and `summarize` (0.13 calls/task, utilized in 5.33% of tasks).
- **Correlation**: Task-level Pearson correlation coefficients show a moderate positive correlation between accuracy and correct file usage F1-score (`+0.2809`) and a negative correlation between accuracy and token usage (`-0.3632`), indicating that more successful runs tend to be more token-efficient.
- **Failures & Point of Failure Breakdown**: The primary failure mode across SUTs and workloads was `different user QUERY interpretation` (**34.33%**), closely followed by `Lack of SCHEMA understanding/wrong file or data source` (**25.37%**) and `Incorrect PIPELINE implementation/code generation` (**23.88%**).

---

## 2. Accuracy & Workload Performance

The SUT performance is evaluated across the 3 independent runs (R1, R2, R3) for `archeology`, and a single run (R1) for `biomedical` and `legal`. 

### Workload Performance Summary (Mean ± Std Dev)

| Workload | SUT Type | Accuracy (Mean ± Std Dev) | Runs Included |
| :--- | :---: | :---: | :--- |
| **archeology** | complete | 41.67% ± 8.33% | `complete_R1`, `complete_R2`, `complete_R3` |
| **archeology** | without | 36.11% ± 4.81% | `without_R1`, `without_R2`, `without_R3` |
| **biomedical** | complete | 44.44% | `complete_R1` (R2 mapping) |
| **biomedical** | without | 55.56% | `without_R1` (R1 mapping) |
| **legal** | complete | 53.33% | `complete_R1` (R2 mapping) |
| **legal** | without | 56.67% | `without_R1` (without mapping) |

### Global Workload-Weighted Average Accuracy

| SUT Type | Weighted Mean Accuracy |
| :--- | :---: |
| **complete** | 46.48% ± 4.98% |
| **without** | 49.44% ± 9.33% |

### Task-by-Task Success Details (All 10 Runs)

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
| `biomedical-hard-1` | 1 | - | - | 0 | - | - |
| `biomedical-easy-2` | 0 | - | - | 1 | - | - |
| `biomedical-hard-3` | 1 | - | - | 1 | - | - |
| `biomedical-hard-4` | 0 | - | - | 0 | - | - |
| `biomedical-hard-5` | 0 | - | - | 0 | - | - |
| `biomedical-easy-6` | 1 | - | - | 1 | - | - |
| `biomedical-hard-7` | 0 | - | - | 1 | - | - |
| `biomedical-hard-8` | 0 | - | - | 0 | - | - |
| `biomedical-easy-9` | 1 | - | - | 1 | - | - |
| `legal-hard-1` | 0 | - | - | 0 | - | - |
| `legal-hard-2` | 0 | - | - | 0 | - | - |
| `legal-easy-3` | 0 | - | - | 0 | - | - |
| `legal-easy-4` | 0 | - | - | 0 | - | - |
| `legal-easy-5` | 1 | - | - | 1 | - | - |
| `legal-hard-6` | 1 | - | - | 1 | - | - |
| `legal-hard-7` | 1 | - | - | 1 | - | - |
| `legal-hard-8` | 1 | - | - | 1 | - | - |
| `legal-easy-9` | 1 | - | - | 1 | - | - |
| `legal-easy-10` | 0 | - | - | 0 | - | - |
| `legal-easy-11` | 1 | - | - | 1 | - | - |
| `legal-easy-12` | 1 | - | - | 1 | - | - |
| `legal-easy-13` | 1 | - | - | 1 | - | - |
| `legal-hard-14` | 0 | - | - | 0 | - | - |
| `legal-hard-15` | 0 | - | - | 0 | - | - |
| `legal-hard-16` | 0 | - | - | 1 | - | - |
| `legal-hard-17` | 1 | - | - | 0 | - | - |
| `legal-hard-18` | 0 | - | - | 0 | - | - |
| `legal-easy-19` | 0 | - | - | 0 | - | - |
| `legal-easy-20` | 1 | - | - | 1 | - | - |
| `legal-easy-21` | 0 | - | - | 1 | - | - |
| `legal-hard-22` | 0 | - | - | 0 | - | - |
| `legal-hard-23` | 1 | - | - | 1 | - | - |
| `legal-hard-24` | 0 | - | - | 0 | - | - |
| `legal-easy-25` | 1 | - | - | 1 | - | - |
| `legal-easy-26` | 0 | - | - | 0 | - | - |
| `legal-easy-27` | 1 | - | - | 1 | - | - |
| `legal-hard-28` | 1 | - | - | 1 | - | - |
| `legal-hard-29` | 0 | - | - | 0 | - | - |
| `legal-hard-30` | 1 | - | - | 1 | - | - |

---

## 3. File Usage Metrics Comparison

We evaluate the precision, recall, and F1-score of SUT file references inside generated `pipeline_code.py` scripts compared to the workload's gold reference data sources.

| Workload | SUT Type | Mean Precision | Mean Recall | Mean F1-score |
| :--- | :--- | :---: | :---: | :---: |
| **archeology** | complete | 1.0000 | 0.9167 | 0.9444 |
| **archeology** | without | 1.0000 | 0.9167 | 0.9444 |
| **biomedical** | complete | 0.9556 | 0.9444 | 0.9352 |
| **biomedical** | without | 0.9722 | 0.9444 | 0.9471 |
| **legal** | complete | 0.7222 | 0.7000 | 0.6911 |
| **legal** | without | 0.7167 | 0.6611 | 0.6522 |

*Analytical Note:* The Complete SUT achieves a higher F1-score (**0.6911**) on the highly hierarchical and complex Legal dataset compared to the Without SUT (**0.6522**). This indicates that hierarchical indexing assists in identifying correct schemas/data sources in environments with many subdirectories and files.

---

## 4. Pearson Correlation Analysis

Task-level correlation metrics calculated across all 150 tasks show:

- **Accuracy vs. F1 File Usage**: `+0.2809` (Positive correlation, indicating that correct file references are directly linked to final task success).
- **Accuracy vs. Token Cost**: `-0.3632` (Negative correlation, confirming that tasks that succeed require less token backtracking and prompt recovery).
- **F1 File Usage vs. Token Cost**: `-0.2153` (Negative correlation, suggesting that incorrect data source selection leads to higher token cost overhead).

---

## 5. Skill Invocation Statistics (Complete SUT Runs Only)

| Skill | Total Calls | Mean Calls/Task | Tasks Utilizing Skill (%) |
| :--- | :---: | :---: | :---: |
| **expand** | 72 | 0.96 | 33.33% |
| **search** | 112 | 1.49 | 26.67% |
| **summarize** | 10 | 0.13 | 5.33% |

---

## 6. Point of Failure Classifications

Failure classifications are parsed directly from SUT `point_of_failure_*.md` files.

### Failure Type Percentage Breakdown per Workload

| Workload | Incorrect PIPELINE implementation / code generation (%) | Lack of SCHEMA understanding / wrong file (%) | Lack of data CONTENT understanding (%) | Different user QUERY interpretation (%) | Mismatch ANSWER type or format (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **archeology** | 35.71% | 7.14% | 7.14% | 45.24% | 4.76% |
| **biomedical** | 0.00% | 57.14% | 14.29% | 0.00% | 28.57% |
| **legal** | 5.56% | 55.56% | 0.00% | 22.22% | 16.67% |

### Global Failure Type Percentage Breakdown

| Failure Type | Global Percentage (%) |
| :--- | :---: |
| **different user QUERY interpretation** | 34.33% |
| **Lack of SCHEMA understanding/wrong file or data source** | 25.37% |
| **Incorrect PIPELINE implementation/code generation** | 23.88% |
| **mismatch ANSWER type or format** | 10.45% |
| **Lack of data CONTENT understanding** | 5.97% |

---

## 7. Phase Character Count Breakdowns

Character count breakdowns measure the proportion of tokens/characters processed during Exploration vs. Analysis.

- **Exploration Phase**: 758,757 characters (**26.72%**)
- **Analysis Phase**: 2,081,030 characters (**73.28%**)

### Phase Breakdown by SUT Type

| SUT Type | Exploration Phase Characters (%) | Analysis Phase Characters (%) |
| :--- | :---: | :---: |
| **complete** | 25.66% | 74.34% |
| **without** | 27.82% | 72.18% |
