# 23CSE301 Machine Learning Capstone - Reviews 1 and 2

The project includes diamond Regression, rice Classification Parts A/B and marketing-customer Clustering, with controlled comparisons within each track. Review 1 has now been audited and repaired as well as Review 2. The source of requirements is the seven-page capstone guidelines PDF in this folder. See the [Review 1 audit](r1/audit/REVIEW1_AUDIT.md), [Review 2 audit](r2/audit/REVIEW2_AUDIT.md) and [website requirements](app/WEBSITE_REQUIREMENTS.md) before submitting.

## Datasets and problem statements

| Track | Assigned file | Problem |
|---|---|---|
| Regression | `data/diamonds_sampled.csv` | Predict `price` from physical measurements and quality categories. Assigned sample: 2,100 rows, 11 raw columns. Identifier `id: 0` is excluded. |
| Classification | `data/riceClassification_sampled.csv` | Predict `Class` from grain measurements. Supplied instructor sample: 1,900 rows, 12 original columns, two classes. `id` is excluded. |
| Clustering | `data/marketing_campaign.csv` | Discover customer groups from demographics and purchasing behavior without training on campaign responses. Supplied file: 2,240 rows, 29 original columns, tab-delimited. |

The supplied datasets are retained under `r2/` as well. Canonical runs prefer `data/`; edit that copy deliberately if replacing a dataset. Use the exact instructor-assigned rice sample rather than downloading a different rice dataset. Dataset licenses and original upstream attribution were not supplied; the team should add these if required by the instructor.

## Setup and execution

Validated environment: Python 3.14, with exact installed dependency versions in `requirements.txt`. Create an environment, install dependencies and launch JupyterLab from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open and run every cell in this order:

1. `notebooks/regression.ipynb`: full diamond audit/EDA, all ten regressors, tuning, R²/RMSE/MAE comparison, top-two five-fold R², coefficients/sparsity, importance and residual/predicted-versus-actual diagnostics.
2. `notebooks/classification.ipynb`: data audit, EDA, all ten baseline/tuned classifiers, five metrics, confusion matrices, ROC curves, full comparison, training-CV selection, top-two CV and final saved pipeline.
3. `notebooks/clustering.ipynb`: data audit/EDA, common preprocessing, K-Means elbow, three hierarchical linkages/dendrograms, mandatory metrics, PCA and t-SNE for both methods, profiles and five-fold refit/stability checks.

The repaired per-algorithm notebooks remain in `r2/classific/` and `r2/clust/`. Run K-Means before the separate Agglomerative notebook. The separate consolidated classification notebook requires all ten summaries from the combined classification notebook; it refuses incomplete or inconsistent inputs. No manual result-file copying is needed.

To execute all ten Review 2/combined notebooks and save visible outputs:

```powershell
$env:REVIEW2_JOBS = '1'
python r2/run_review2.py
```

The runner directs Jupyter/IPython runtime files into the writable workspace. In a restricted environment where starting a kernel is unavailable, the verified fallback executes the actual cells using IPython with captured tables, prints and plot images:

```powershell
python r2/run_review2.py --in-process
```

`--only classification` and `--only clustering` run either track. Re-running all notebooks repeats the individual model fits by design. `REVIEW2_JOBS` controls parallel tuning (default 2); 1 avoids unnecessary process overhead on small datasets. Full original notebook backups: `r2/audit/original_notebooks.zip`.

## Evaluation policy

All ten classifiers use the same cleaned sample, engineered Compactness feature, stratified 80:20 split and seed 42. Imputation, IQR clipping and scaling are inside the model pipeline and are refitted inside every tuning fold. All grids use five-fold stratified CV on training rows. The split manifest records the dataset hash, features and exact train/test indices. Summaries from different splits are rejected.

The table reports tuned models with accuracy, weighted precision, weighted recall, weighted F1 and probability-based ROC-AUC (weighted OvR for multiclass). Baseline-versus-tuned changes are saved separately. Selection uses training CV weighted F1, not the test-set ranking. Five-fold results for the top two reuse the tuning folds, so they are not an unbiased nested-CV estimate; the reserved test set provides the final performance report.

Clustering uses one unlabeled cohort and the same preprocessing for both algorithms. It excludes IDs, constant fields and campaign outcomes. Age is referenced to the observed collection period; impossible ages become missing. Total spend/purchases and tenure summarize behavior. Numeric IQR clipping limits extreme influence, median imputation handles missing values and one-hot encoding handles categoricals. These are explicit scaffolding choices for the team to review. K is selected from the elbow's geometric knee. Hierarchical linkage selection requires every cluster to contain at least 2% of customers, then maximizes silhouette. All three internal metrics and cluster sizes remain visible, including rejected tiny-cluster solutions. PCA/t-SNE are projections, not the fitting space.

Five-fold clustering subset refits independently fit preprocessing and compare metrics and adjusted Rand stability to the full fit. This is unsupervised resampling validation, not supervised prediction CV; Agglomerative clustering has no native held-out `predict`. Campaign responses are inspected only after fitting. Confirm with the instructor if a different interpretation of the general cross-validation guideline is expected for clustering.

## Results summary

<!-- RESULTS_START -->

| CV_rank | Algorithm | Accuracy | Precision | Recall | F1_weighted | ROC_AUC | CV_F1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Support Vector Machine | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.995700 | 0.991445 |
| 2 | Logistic Regression | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.995951 | 0.990788 |
| 3 | K-Nearest Neighbors | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.989514 | 0.990785 |
| 4 | Gradient Boosting | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.994345 | 0.989474 |
| 5 | AdaBoost | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.994583 | 0.988161 |
| 6 | Random Forest | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.994052 | 0.988157 |
| 7 | MLP | 0.984211 | 0.984655 | 0.984211 | 0.984184 | 0.993661 | 0.988151 |
| 8 | Bagging | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.991720 | 0.986183 |
| 9 | Naive Bayes | 0.984211 | 0.984655 | 0.984211 | 0.984184 | 0.995616 | 0.984203 |
| 10 | Decision Tree | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.988090 | 0.983551 |

CV-selected final model: **Support Vector Machine** (selection uses training CV weighted F1, not test ranking). ROC_AUC: 0.993829 → 0.995700 (change +0.001871)

| Algorithm | Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| --- | --- | --- | --- | --- |
| K-Means | 3 | 0.190328 | 1.812027 | 530.712431 |
| Agglomerative | 3 | 0.173385 | 1.906311 | 473.794804 |

All ten executable Review 2 notebooks passed actual cell execution with visible outputs. Five focused tests passed; all 50 classification metrics and six clustering metrics were independently recomputed from saved predictions/assignments. See `r2/audit/final_verification.json`.

<!-- RESULTS_END -->

Outputs include `r2/results/classification_all_10_comparison.csv`, `classification_top2_cv.csv`, `classification_split.json`, per-model search results/predictions/tuning changes, and `clustering_comparison.csv`, linkage comparison, resampling tables, profiles, assignments and plots. Complete fitted classifier pipelines are in `models/`; `final_classifier.joblib` is the CV-selected model. Load trusted local artifacts only. K-Means is also saved with its fitted preprocessing; hierarchical clustering is transductive and is represented by its saved assignments and fitting configuration.

## Required team completion

The code and measured results do not complete the in-person/team requirements. Replace the marked interpretation cells with the team's original observations, write the final justification using actual metrics/plots, and assign each team member primary ownership of a track. Rehearse the problem → data → method → results → conclusion narrative and metric/algorithm trade-offs.

There was no Git repository or configured GitHub remote in the supplied folder. Publish to the team's GitHub repository and preserve genuine meaningful milestone commits; the historical requirement cannot be recreated by inventing a backdated history. GUI/public deployment are optional bonus items. The reference-led website is implemented under `app/site/`, with the three-track lab, verified data/plots and real local browser inference. Following explicit user approval, private deployment succeeded on 19 October 2026: [Open Cipher Pol Learning Lab](https://ordo-learning-lab.tenderlamp10.chatgpt.site). Access remains restricted to the owner account; no public-deployment bonus is claimed. Deployment details are recorded in `app/hosting_deployment.json`. Review 1 now has repaired full regression and Part A notebooks; their original versions are archived under `r1/audit/`.

## Review 1 results and execution

<!-- REVIEW1_START -->

| Rank_by_test_R2 | Model | R2 | RMSE | MAE | CV_R2_Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | Support Vector Regression | 0.944879 | 972.320440 | 519.597611 | 0.951683 |
| 2 | Lasso Regression | 0.935244 | 1053.873517 | 696.546805 | 0.934319 |
| 3 | Linear Regression | 0.935195 | 1054.271075 | 697.477719 | 0.934274 |
| 4 | Ridge Regression | 0.935157 | 1054.579628 | 696.185465 | 0.934413 |
| 5 | ElasticNet Regression | 0.935126 | 1054.834351 | 695.331291 | 0.934441 |
| 6 | Polynomial Regression | 0.934149 | 1062.744297 | 692.804134 | 0.914148 |
| 7 | Gradient Boosting Regression | 0.930859 | 1088.975294 | 511.795296 | 0.943171 |
| 8 | Random Forest Regression | 0.924145 | 1140.622824 | 549.306027 | 0.935740 |
| 9 | KNN Regression | 0.908780 | 1250.820088 | 668.232271 | 0.916491 |
| 10 | Decision Tree Regression | 0.898813 | 1317.379766 | 656.527196 | 0.895898 |

Selected regression model by training CV R²: **Support Vector Regression**. Five-fold CV rows and explicit baseline/tuned changes are saved in `r1/results/`. Price currency is not verified from the supplied metadata.

Review 1 Part A:

| Algorithm | Accuracy | Precision | Recall | F1_weighted | ROC_AUC |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | 0.984211 | 0.984655 | 0.984211 | 0.984184 | 0.995616 |
| Logistic Regression | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.995951 |
| K-Nearest Neighbors | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.989514 |
| Decision Tree | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.988090 |
| Support Vector Machine | 0.981579 | 0.981873 | 0.981579 | 0.981554 | 0.995700 |

Verified: **18 executed Review 1/combined notebooks**, **30 independently recomputed regression metrics**, five-fold R² for the final top two, fold-specific fitted preprocessing, selected model reload, **21 saved-model inference parity checks** (10 regression + 10 classification + K-Means), and invalid-input rejection. Full record: `r1/audit/final_verification.json`. Team-written interpretation/justification, real GitHub history and in-person presentation/viva remain outstanding.

<!-- REVIEW1_END -->

Run all repaired Review 1 notebooks with visible outputs:

```powershell
$env:REVIEW2_JOBS = '1'
python r2/run_review2.py --review1 --in-process
```

Current regression inputs are shared across all models: identifier removal, duplicates, invalid-dimension handling, volume, price-decile-stratified 80:20 split and training-fold median/IQR/scaling/one-hot preprocessing. PolynomialFeatures expands numeric predictors only (degrees 2/3). SVR also fits a target scaler within training folds and inverse-transforms predictions into original price units. Rank the report by held-out R²; select `models/final_regressor.joblib` using training CV R². Root `models/final_classifier.joblib` remains selected by training CV weighted F1. Archival CSVs under the old nested Review 1 result folders must not supply the website's current values.

Website preparation: `python app/export_site_data.py` rebuilds the public result/plot catalogue from canonical outputs; `python r1/audit/verify_review1.py` checks the repaired notebooks, saved metrics and all 21 supported inference paths. `app/inference.py` provides the Python reference functions. The website in `app/site/` evaluates exported fitted models in JavaScript, avoiding an external Python service. All 10,058 parity checks passed; see `app/site/model-parity-report.json`. Launch locally with `node app/site/server.mjs`.

## AI assistance and sources

OpenAI Codex assisted with the requirement audit, code scaffolding, bug fixes, notebook restructuring, execution checks and documentation. In accordance with guideline section 7.5, team analysis and interpretation have deliberately been left for the team, visibly marked in notebook Markdown cells. No performance values have been invented. No external code snippets were copied. The implementation uses the installed scikit-learn, NumPy, pandas, SciPy, Matplotlib, Seaborn, IPython and Jupyter libraries.

## Repository layout

```text
README.md
requirements.txt
data/                         assigned input datasets
notebooks/                    complete regression/classification/clustering tracks
r1/                           repaired Review 1 notebooks, results and original backups
r2/review2_pipeline.py         shared fitting/evaluation functions
r2/classific/                 five individual Part B notebooks
r2/clust/                     two individual clustering notebooks
r2/results/                   measured tables and figures
r2/audit/                     requirement audit, backup, execution and test records
models/                       locally generated trained pipelines
app/                          website requirements, inference and verified data/plot export
```

Focused verification: `python r2/test_review2.py`. These tests use temporary synthetic examples only to check software behavior; submitted results always come from the assigned datasets.

## Cipher Pol website verification — 19 October 2026

The private website is branded Cipher Pol. Its final verification covers all 22 algorithms, 135 fresh supervised search configurations (675 five-fold CV fits), both clustering methods and stability checks, 80 held-out metrics recomputed from the website JSON, 60 source-identical figures, and all three executed notebook/CSV downloads. Browser inference passed 45,267 all-data and stress checks; 698 visible metric cells and all 21 prediction forms were checked against the fitted sklearn models. Maximum price error was 2.37e-10 dataset units; class and cluster assignments and all displayed two-decimal prices matched. Full measured evidence: `app/deep-verification.json`.

The deep check fixed numeric export rounding and missing-derived-feature imputation, and replaced KNN regression brute-force distance search with deterministic kd-tree search. Regression notebooks, model artifacts, CV results, plots and website downloads were refreshed together. These checks verify implementation and measured results on the supplied datasets; they do not imply perfect predictions for unseen records.

The model-detail check also restricts odds ratios to logistic regression. Linear SVM coefficients are shown as coefficients, without an invalid odds-ratio interpretation.
