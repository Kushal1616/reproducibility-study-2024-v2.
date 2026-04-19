# Granular and Type-2 Fuzzy Collaborative Learning with Large and Small Models for Industrial Time-Series Prediction

This repository accompanies the implementation of the paper:

**Granular and Type-2 Fuzzy Collaborative Learning with Large and Small Models for Industrial Time-Series Prediction**.

It provides scripts for reproducing the main experimental tables, preparing CMAPSS splits, tuning the proposed model, and running ablation and multi-horizon conformal experiments.

## Dataset

The experiments use the **Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset, a standard benchmark for **remaining useful life (RUL)** prediction of turbofan engines.

### CMAPSS overview

- Multivariate time-series benchmark for prognostics.
- Each engine trajectory evolves under operating conditions until failure.
- Each cycle contains:
  - **3 operational settings**
  - **21 sensor measurements**
- The dataset is divided into four subsets:
  - **FD001**
  - **FD002**
  - **FD003**
  - **FD004**

These subsets differ in operating conditions and fault modes. The implementation follows the standard preprocessing and train-test setup commonly used in the CMAPSS literature.

## Required libraries

### Core scientific stack

- `numpy`
- `pandas`
- `scikit-learn`

Used mainly for:
- `mean_squared_error`
- `mean_absolute_error`
- `StandardScaler` in selected scripts

### Deep learning

- `torch`

Commonly used modules include:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `torch.utils.data`

### LLM / pretrained backbone support

- `transformers`

This is specifically required by `paper8_fd001_ablation.py` for:
- `GPT2Model`
- `GPT2Config`

That script explicitly aborts if `transformers` is unavailable unless an unpretrained fallback is allowed.

### Python standard-library modules used

These do not require separate installation, but they are used by the scripts:

- `os`
- `json`
- `math`
- `time`
- `copy`
- `random`
- `argparse`
- `importlib`
- `importlib.util`
- `glob`
- `pathlib`
- `dataclasses`
- `typing`
- `sys`

## Installation

Install the main dependencies with:

```bash
pip install numpy pandas scikit-learn torch transformers
```

## Code-to-paper mapping

The following table maps the main scripts to the paper outputs, dataset usage, and file access patterns.

| Code file | Best paper mapping | Type of output generated | Dataset used | Where it reads data from | Main output / note |
|---|---|---|---|---|---|
| `unified_earlystop_comparison_runner.py` | Table III | Unified early-stop comparison for non-LLM baselines plus CoLLM-C and GF-CoLLM; reports `Val_RMSE_best`, `Test_RMSE`, `Test_MAE`, `Runtime_sec`, `Status` | Any FD subset passed in | Reads `train_{FD}.txt`, `test_{FD}.txt`, `RUL_{FD}.txt` from `data_dir`; uses split JSON; imports `paper7_{fd}` and `paper8_{fd}` | Best match for the combined comparison table structure in Table III |
| `unified_earlystop_comparison_runner_fd003_recipe.py` | Table III (FD003-focused variant) | Same type of unified comparison output as above, adapted to FD003 recipe | CMAPSS FD003 | Uses `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt` through `data_dir` and split JSON | Support / dataset-specific Table III runner |
| `unified_earlystop_comparison_runner_fd004_dualexpert_final.py` | Table III (FD004-focused variant) | Same unified comparison output; FD004-specific module fallback and dual-expert recipe support | CMAPSS FD004 | Uses `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt` through `data_dir` and split JSON | Support / dataset-specific Table III runner |
| `unified_llm_comparison_runner_fd001.py` | Table III | Uniform seeded early-stop benchmark for GPT-2, Llama-2-7B, OneFitsAll-FPT, AutoTimes, Qwen2.5-0.5B, Qwen3-0.6B, optional CoLLM-C, GF-CoLLM; outputs CSV with dataset/seed/group/model/RMSE/MAE/runtime/status | CMAPSS FD001 | Reads `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` and split JSON | Strong Table III match for FD001 |
| `unified_llm_comparison_runner_fd002.py` | Table III | Same as above for FD002 | CMAPSS FD002 | Reads `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt` and split JSON | Strong Table III match for FD002 |
| `unified_llm_comparison_runner_fd003.py` | Table III | Same as above for FD003 | CMAPSS FD003 | Reads `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt` and split JSON | Strong Table III match for FD003 |
| `unified_llm_comparison_runner_fd004.py` | Table III | Same as above for FD004 | CMAPSS FD004 | Reads `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt` and split JSON | Strong Table III match for FD004 |
| `fd001_branch_value_table_clean.py` | Table IV (FD001 block) | Routing-mode table with `rmse`, `mae`, `coverage`, `interval_width`, `latency_ms_per_sample` for `small_only`, `corrected_large_side`, `threshold_routing`, `mlp_gate`, `type1_fuzzy_routing`, `it2_fuzzy_routing`, `gf_collm` | CMAPSS FD001 | `./FD001/train_FD001.txt`, `./FD001/test_FD001.txt`, `./FD001/RUL_FD001.txt` | Writes `fd001_branch_value_outputs/fd001_branch_value_table_clean.csv` |
| `fd002_branch_value_table_clean.py` | Table IV (FD002 block) | Same Table IV routing-mode outputs | CMAPSS FD002 | `./FD002/train_FD002.txt`, `./FD002/test_FD002.txt`, `./FD002/RUL_FD002.txt` | Writes `fd002_branch_value_outputs/fd002_branch_value_table_clean.csv` |
| `fd003_branch_value_table_clean.py` | Table IV (FD003 block) | Same Table IV routing-mode outputs | CMAPSS FD003 | `./FD003/train_FD003.txt`, `./FD003/test_FD003.txt`, `./FD003/RUL_FD003.txt` | Writes `fd003_branch_value_outputs/fd003_branch_value_table_clean.csv` |
| `fd004_branch_value_table_clean.py` | Table IV (FD004 block) | Same Table IV routing-mode outputs | CMAPSS FD004 | `./FD004/train_FD004.txt`, `./FD004/test_FD004.txt`, `./FD004/RUL_FD004.txt` | Writes `fd004_branch_value_outputs/fd004_branch_value_table_clean.csv` |
| `multihorizon_h5_splitcp_baselines_tuned_with_gfcollm_fixed.py` | Table V for H=5 | Multi-horizon conformal comparison across GRU + Split CP, Transformer + Split CP, CoLLM-C + Split CP, GF-CoLLM + Split CP; summary metrics include `Avg_RMSE`, `Final_Horizon_RMSE`, `Coverage`, `Avg_Interval_Width` | User-selected CMAPSS FD subset | Reads `train_{FD}.txt`, `test_{FD}.txt`, `RUL_{FD}.txt` from `data_dir`; uses split JSON | Best direct Table V match for H=5 |
| `multihorizon_fd002_splitcp_tuned.py` | Table V (FD002) | Same Table V multi-method outputs | CMAPSS FD002 | Default `FD002/train_FD002.txt`, `FD002/test_FD002.txt`, `FD002/RUL_FD002.txt` | Summary CSV for FD002 multi-horizon Split CP |
| `multihorizon_fd003_splitcp_tuned.py` | Table V (FD003) | Same Table V multi-method outputs | CMAPSS FD003 | Reads `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt` | Summary CSV for FD003 multi-horizon Split CP |
| `multihorizon_fd004_splitcp_tuned.py` | Table V (FD004) | Same Table V multi-method outputs | CMAPSS FD004 | Default `FD004/train_FD004.txt`, `FD004/test_FD004.txt`, `FD004/RUL_FD004.txt` | Summary CSV for FD004 multi-horizon Split CP |
| `combined_h5_coverage_all_fd.py` | Table V-style support | LSTM-based horizon-wise MAE/RMSE/coverage/width plus summary over all FDs | CMAPSS FD001-FD004 | `FD001/`, `FD002/`, `FD003/`, `FD004/` folders with train/test/RUL files | Support script, not exact named Table V method set |
| `combined_h10_coverage_all_fd.py` | Table V-style support | Same as above for H=10 | CMAPSS FD001-FD004 | Same four folders | Writes `coverage_outputs_h10/...` |
| `combined_h20_coverage_all_fd.py` | Table V-style support | Same as above for H=20 | CMAPSS FD001-FD004 | Same four folders | Writes `coverage_outputs_h20/...` |
| `paper8_fd001_ablation.py` | FD001 ablation / edge-cloud analysis | Edge-only vs cloud-only vs hybrid; reports latency, memory footprint, large-model call rate, RMSE/MAE in cycles; conformal confidence; pretrained GPT-2 backbone | CMAPSS FD001 | In `--data_dir`: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` | Core ablation implementation, not Table III/IV/V |
| `fd_001big_ablation.py` | Wrapper for FD001 ablation | Runner only | CMAPSS via `--data_dir` | User-supplied CMAPSS path | Calls `paper8_fd001_ablation.py` |
| `prepare_cmapss_splits.py` | Support only | Creates deterministic train/validation engine split JSON | Any chosen FD subset | Reads `train_{FD}.txt` from `data_dir` | Writes `splits/{FD}_seed{seed}.json` |
| `tune_gfcollm_runner.py` | Support only | Tunes GF-CoLLM schedules/hyperparameters; evaluates branch component RMSEs | FD-aware | Reads `train_{FD}.txt`, `test_{FD}.txt`, `RUL_{FD}.txt` through `paper8_{fd}` loaders and split JSON | Tuning support |
| `tune_gfcollm_runner_FD001.py` | Support only | Tunes FD001 GF-CoLLM | CMAPSS FD001 | Via `paper8_fd001.py` loaders and FD001 files | Tuning support |
| `tune_gfcollm_runner_fd003_v3.py` | Support only | Tunes FD003 GF-CoLLM | CMAPSS FD003 | Reads FD003 files | Tuning support |
| `tune_gfcollm_runner_fd004_v12_dualfix.py` | Support only | Tunes FD004 GF-CoLLM dual-expert recipe | CMAPSS FD004 | Reads FD004 files | Tuning support |

## Best direct matches by paper item

### Table III

The strongest matches for **Table III** are:

- `unified_earlystop_comparison_runner.py`
- `unified_earlystop_comparison_runner_fd003_recipe.py`
- `unified_earlystop_comparison_runner_fd004_dualexpert_final.py`
- `unified_llm_comparison_runner_fd001.py`
- `unified_llm_comparison_runner_fd002.py`
- `unified_llm_comparison_runner_fd003.py`
- `unified_llm_comparison_runner_fd004.py`

These are the best matches because Table III is the combined comparison with non-LLM and higher-capacity / collaborative baselines across **FD001-FD004**, reported mainly in **RMSE** and **MAE**, and these runners generate exactly that style of benchmark CSV.

### Table IV

The strongest matches for **Table IV** are:

- `fd001_branch_value_table_clean.py`
- `fd002_branch_value_table_clean.py`
- `fd003_branch_value_table_clean.py`
- `fd004_branch_value_table_clean.py`

These are nearly one-to-one matches with the routing-mode structure reported in Table IV.

### Table V

The strongest matches for **Table V** are:

- `multihorizon_h5_splitcp_baselines_tuned_with_gfcollm_fixed.py`
- `multihorizon_fd002_splitcp_tuned.py`
- `multihorizon_fd003_splitcp_tuned.py`
- `multihorizon_fd004_splitcp_tuned.py`

These are the strongest Table V matches because they implement the named methods and conformal metrics used in the paper's multi-horizon comparison.

### Support-only scripts

The following scripts mainly support reproducibility and tuning, but they are not direct paper tables or figures:

- `prepare_cmapss_splits.py`
- all `tune_gfcollm_runner*.py` files
- `fd_001big_ablation.py`

## Paper numbering and implementation notation

| Paper No. | Paper Title | Implementation Notation |
|---|---|---|
| Paper 1 | Language Models are Unsupervised Multitask Learners | `Paper1_fd001` to `Paper1_fd004` |
| Paper 2 | Llama 2: Open Foundation and Fine-Tuned Chat Models | `Paper2_fd001` to `Paper2_fd004` |
| Paper 3 | One Fits All: Power General Time Series Analysis by Pretrained LM | `Paper3_fd001` to `Paper3_fd004` |
| Paper 4 | AutoTimes: Autoregressive Time Series Forecasters via Large Language Models | `Paper4_fd001` to `Paper4_fd004` |
| Paper 5 | Qwen2.5 Technical Report | `Paper5_fd001` to `Paper5_fd004` |
| Paper 6 | Qwen3 Technical Report | `Paper6_fd001` to `Paper6_fd004` |
| Paper 7 | CoLLM: Industrial Large-Small Model Collaboration with Fuzzy Decision-making Agent and Self-Reflection | `Paper7_fd001` to `Paper7_fd004` |
| Paper 8 | Granular and Type-2 Fuzzy Collaborative Learning with Large and Small Models for Industrial Time-Series Prediction (Proposed Paper) | `Paper8_fd001` to `Paper8_fd004` |

## Notes

- The repository is centered on **CMAPSS-based industrial time-series prediction**.
- The direct paper reproduction is mainly organized around:
  - **Table III**: unified comparison scripts
  - **Table IV**: branch-value / routing-mode scripts
  - **Table V**: multi-horizon conformal comparison scripts
- Split preparation and tuning utilities are included to support consistent reproducibility across seeds and FD subsets.
