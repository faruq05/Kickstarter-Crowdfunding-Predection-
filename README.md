# Kickstarter-Crowdfunding-Predection-

# Kickstarter Campaign Funding Prediction

A regression study predicting the final USD amount raised by Kickstarter campaigns (`log_target = log1p(target_usd)`), comparing five independent modeling paradigms — classical tabular ML, GBDT stacking, NLP embeddings, unsupervised clustering, and LLM fine-tuning — evaluated on a common 9-metric suite. Built as a CSE445 (NSU) machine learning course project.

## Problem Statement

Given structured campaign metadata (goal, category, timing, text length, etc.), predict how much money a Kickstarter campaign will raise — a **regression** task, not the more commonly studied success/failure classification.

## Dataset

- **Raw source**: merged Kickstarter campaign snapshot CSVs (`Kickstarter_raw*.csv`), ~20,000 completed campaigns (successful/failed/canceled only; live/ongoing campaigns excluded since final funding is unknown).
- **Files**:
  - `kickstarter_raw_before_encoding.csv` — cleaned, pre-encoding (20,000 rows × 25 cols)
  - `ML_train.csv` / `ML_test.csv` — final encoded 80/20 split (77 features)
- **Target**: `target_usd` (raw) and `log_target` (log1p-transformed, used for training/evaluation).

## Diagram
<img width="4339" height="2081" alt="malware 499 course" src="https://github.com/user-attachments/assets/f709e85a-7375-4303-921d-f83428a15823" />


## Preprocessing Pipeline

**Stage 1 — Cleaning (`kickstarter_preprocessing_stage1.ipynb`)**
1. Merge multi-file raw CSVs, check schema consistency
2. Deduplicate campaigns (keep latest by `state_changed_at`)
3. Filter to completed campaigns only
4. Build regression target (`target_usd`, `log_target`)
5. Parse nested JSON fields → `category_parent`, `category_name`, `location_country`, `location_state`, `location_type`
6. Convert goal to USD (`goal_usd`, `log_goal_usd`)
7. Engineer text-length features (name/blurb char & word counts)
8. Engineer video indicator, prelaunch-activated flag
9. Convert Unix timestamps → `duration_days`, `prelaunch_days`, `launch_year/month/day/weekday/hour`
10. Drop target-leakage columns (`pledged`, `backers_count`, `state`, `spotlight`, etc.)
11. Drop irrelevant/redundant columns (raw JSON, raw text, media objects, currency metadata)
12. Cardinality inspection for encoding strategy

**Stage 2 — Encoding & Split (`kickstarter_encoding_train_test_single_files.ipynb`)**
1. Stratified 80/20 train/test split (on quantile-binned `log_target`)
2. Median imputation (fit on train only)
3. Frequency encoding for high-cardinality categoricals (fit on train only)
4. One-hot encoding for low-cardinality categoricals (`drop="first"`, fit on train only)
5. Consistency checks (no leakage, no NaNs, matching train/test columns) → final `ML_train.csv` / `ML_test.csv`

All fitted transforms (medians, frequency maps, encoder) are fit strictly on training data to prevent leakage.

## Modeling Tracks

### Track A — Tabular ML (`kickstarter_all_models_ensembles_final.ipynb`, `kickstarter_all_ensembles_models_tuned_3_methods.ipynb`)
- 9 base models: RF, Extra Trees, Decision Tree, AdaBoost, XGBoost, CatBoost, KNN, Ridge, SVR
- 3 ensemble methods: Stacking, Blending, Bagging
- 3 tuning strategies: RandomizedSearchCV, HalvingGridSearchCV, BayesSearchCV
- **Best**: Stacking (baseline, untuned) — RMSE_log 2.2295, R²_log 0.4709

### Track B — GBDT Stacking Pipeline (`Method_3_Stacking_RF_ET_XGB_CatBoost_SVR_RandomReLU_PCA_MLP_SoftRetrieval.ipynb`)
iLTM-inspired architecture (arXiv:2511.15941): RF+ET+XGBoost+CatBoost+SVR base learners → 5-fold OOF stacking with Ridge meta-learner → concatenate original features + stacking representation → random feature expansion → ReLU → PCA → MLP → soft k-NN retrieval blended with validation-selected α.

### Track C — NLP Embeddings (`kickstarter_nlp_embeddings_creation.ipynb`, `nlp_model_base_tune_colab.ipynb`)
- 5 embedding methods: Word2Vec, SBERT, DistilBERT, BERT, TF-IDF
- 3 downstream models: XGBoost, CatBoost, RF
- 3 tuning methods → 45 total runs
- **Best overall model**: BERT + XGBoost (BayesSearchCV) — RMSLE 1.5097, R²_log 0.757

### Track D — GMM Clustering (`gmm_embedding_baseline.ipynb`)
Gaussian Mixture Model soft clustering on NLP embeddings with cluster-conditioned regression.

### Track E — LLM Fine-Tuning (LoRA)
- Models: Qwen2.5-0.5B-Instruct, TinyLlama-1.1B-Chat
- Generative SFT with masked labels, `apply_chat_template()`, regex numeric parsing
- 4 serialization formats: terse, human_readable, prompt_style, json
- Baseline + Optuna-tuned variants, split across notebooks (`qwen_1_baseline.ipynb` + `qwen_2_optuna.ipynb`, `tinyllama_baseline.ipynb` + `tinyllama_optuna.ipynb`) to fit Kaggle's 12-hour session limit, with `shared_config.json` handoff for identical splits and SQLite-backed Optuna for resumability.

### Ablation Studies (NLP track)
PCA dimensionality, embedding pooling strategy, feature-set comparison, serialization template variants, hyperparameter sensitivity, tuning-method comparison, top-N PCA importance dropout, outlier sensitivity, cross-embedding comparison, CV fold sensitivity, goal_usd inclusion/exclusion, cumulative feature addition, noise robustness, preprocessing variants.

Key finding: **Category** and **Timing** feature groups are the most decisive (ΔRMSLE +0.173 and +0.158), outweighing the goal amount.

### Explainability (XAI)
SHAP (KernelExplainer) and LIME (LimeTabularExplainer) applied to the best model (BERT + XGBoost). A target-leakage bug (`target_usd` inside `serialize_row()`) was found and fixed before final XAI runs; top post-fix drivers: `category_parent`, `category_name`, `location_state`.

## Evaluation Metrics

Every track is scored on the same 9-metric suite:
`MAE`, `MSE`, `RMSE`, `R²` (each computed in both log-space and USD-space), plus `RMSLE`.

## Results Summary

| Family | Best Model | MAE_log | RMSE_log | R²_log | RMSLE |
|---|---|---|---|---|---|
| ML (tabular) | Stacking (baseline) | 1.6463 | 2.2295 | 0.4709 | 2.2294 |
| NLP | BERT + XGBoost (BayesSearchCV) | 1.1168 | 1.5109 | 0.7570 | 1.5097 |
| LLM | Qwen2.5-0.5B LoRA (baseline) | 1.7582 | 2.7896 | 0.1717 | 2.7896 |

**Overall best: NLP track (BERT + XGBoost)** — outperforms all other families on every log-space metric.

## Environment & Tools

- **Local**: VS Code (Windows, GPU)
- **Compute**: Kaggle GPU sessions (12-hour session limit), Google Colab (fallback)
- **Libraries**: scikit-learn, XGBoost, CatBoost, transformers, PEFT/LoRA, Optuna, SHAP, LIME, pandas, seaborn

## Key Learnings

- Category and Timing features dominate over Goal amount in driving funding outcomes.
- Tuning yields only marginal gains over strong baselines; embedding/model choice matters more than tuning method.
- BayesSearchCV consistently outperforms HalvingGridSearchCV and is dramatically faster.
- Always audit serialization/feature pipelines for target leakage before generating embeddings.
- Log-scale metrics (RMSLE, log R²) are essential given the bimodal, right-skewed funding distribution.
