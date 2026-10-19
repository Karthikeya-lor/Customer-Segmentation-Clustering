"""Reproducible Review 2 workflows. Notebook cells explain and display each stage."""
from pathlib import Path
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'r2' / 'results'
MODELS = ROOT / 'models'
for folder in (RESULTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)
SEED = 42
JOBS = int(os.environ.get('REVIEW2_JOBS', '2'))
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1_weighted', 'ROC_AUC']
CLUSTER_METRICS = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
sns.set_theme(style='whitegrid', palette='colorblind')


def data_path(name):
    for path in [ROOT / 'data' / name, ROOT / 'r2' / name]:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Place the instructor-assigned {name} in {ROOT / "data"}.')


def save_plot(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=140, bbox_inches='tight')
    plt.show()
    plt.close('all')


def slug(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def audit_frame(df, target=None):
    print('Shape:', df.shape, '| Exact duplicates:', df.duplicated().sum())
    display(df.head())
    display(pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum()}))
    if target:
        display(df[target].value_counts(dropna=False).to_frame('Rows'))


def eda(df, prefix, target=None):
    """Descriptive plots only; the team must supply its own written interpretation."""
    numeric = df.select_dtypes(include=np.number).drop(columns=['id', 'ID'], errors='ignore')
    axes = numeric.hist(bins=25, figsize=(15, max(6, 2.7 * int(np.ceil(len(numeric.columns) / 4)))), layout=(int(np.ceil(len(numeric.columns) / 4)), 4), color='#0173b2')
    for ax in np.asarray(axes).flat:
        if ax.has_data():
            ax.set_xlabel(ax.get_title()); ax.set_ylabel('Rows')
    plt.suptitle(f'{prefix.title()}: numeric feature distributions', y=1.01)
    save_plot(prefix + '_distributions.png')
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric.corr(), cmap='vlag', center=0, vmin=-1, vmax=1)
    plt.title(prefix.title() + ': feature correlation'); plt.xlabel('Feature'); plt.ylabel('Feature')
    save_plot(prefix + '_correlation.png')
    if target:
        df[target].value_counts().sort_index().plot.bar(color='#0173b2', rot=0)
        plt.title('Rice class distribution'); plt.xlabel('Class'); plt.ylabel('Rows')
        save_plot(prefix + '_target.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, ['Area', 'Perimeter']):
            sns.scatterplot(data=df, x=col, y=target, hue=target, palette='colorblind', alpha=.35, ax=ax)
            ax.set_title(f'{col} versus class')
        save_plot(prefix + '_feature_target.png')


class IQRClipper(TransformerMixin, BaseEstimator):
    """Learn clipping bounds inside each training fold, matching the Review 1 policy."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        self.lower_ = q1 - self.factor * (q3 - q1)
        self.upper_ = q3 + self.factor * (q3 - q1)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def rice_data(show_eda=False):
    path = data_path('riceClassification_sampled.csv')
    raw = pd.read_csv(path)
    audit_frame(raw, 'Class')
    if raw['Class'].isna().any():
        raise ValueError('Missing class labels require instructor clarification.')
    df = raw.drop(columns=['id'], errors='ignore').copy()
    # Deterministic geometry feature from Review 1; no fitted statistics or labels.
    df['Compactness'] = 4 * np.pi * df['Area'] / (df['Perimeter'] ** 2 + 1e-8)
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns='Class'), df['Class']
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError('Unexpected non-numeric rice features; do not silently drop them.')
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED, stratify=y)
    manifest = {'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'feature_names': X.columns.tolist(), 'random_state': SEED,
        'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist()}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['split_signature'] = signature
    (RESULTS / 'classification_split.json').write_text(json.dumps(manifest, indent=2))
    print('After cleaning:', df.shape, '| Train:', len(train_idx), '| Test:', len(test_idx))
    if show_eda:
        eda(df, 'rice', 'Class')
    return {'X': X, 'y': y, 'train': train_idx, 'test': test_idx, 'signature': signature}


def classifier_specs():
    return {
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=SEED), {'C': [.1, 1, 10]}),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [5, 11, 15], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}),
        'Naive Bayes': (GaussianNB(), {'var_smoothing': [1e-10, 1e-9, 1e-8]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {'max_depth': [3, 5, None], 'min_samples_split': [2, 5]}),
        'Support Vector Machine': (SVC(probability=True, random_state=SEED), {'C': [.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Random Forest': (RandomForestClassifier(random_state=SEED, n_jobs=1), {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}),
        'AdaBoost': (AdaBoostClassifier(random_state=SEED), {'n_estimators': [50, 100, 200], 'learning_rate': [.1, .5, 1.]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=SEED), {'n_estimators': [100, 150], 'learning_rate': [.05, .1], 'max_depth': [2, 3]}),
        'Bagging': (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), random_state=SEED, n_jobs=1), {'n_estimators': [50, 100], 'max_samples': [.7, 1.], 'estimator__max_depth': [None, 5]}),
        'MLP': (MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED), {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'activation': ['relu', 'tanh'], 'alpha': [.0001, .001]})
    }


def classification_metrics(model, X, y):
    pred, probs = model.predict(X), model.predict_proba(X)
    auc = roc_auc_score(y, probs[:, 1]) if len(model.classes_) == 2 else roc_auc_score(y, probs, labels=model.classes_, multi_class='ovr', average='weighted')
    return dict(zip(METRICS, [accuracy_score(y, pred), precision_score(y, pred, average='weighted', zero_division=0), recall_score(y, pred, average='weighted', zero_division=0), f1_score(y, pred, average='weighted', zero_division=0), auc]))


def classification_plots(model, X, y, name):
    key = slug(name)
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap='Blues', values_format='d')
    plt.title(name + ': held-out confusion matrix'); save_plot(key + '_confusion.png')
    probs = model.predict_proba(X)
    if len(model.classes_) == 2:
        RocCurveDisplay.from_predictions(y, probs[:, 1], pos_label=model.classes_[1], name=name)
    else:
        binary = label_binarize(y, classes=model.classes_)
        _, ax = plt.subplots()
        for i, cls in enumerate(model.classes_):
            RocCurveDisplay.from_predictions(binary[:, i], probs[:, i], name=f'Class {cls}', ax=ax)
    plt.title(name + ': held-out ROC'); save_plot(key + '_roc.png')


def run_classifier(name, data):
    model, grid = classifier_specs()[name]
    X, y, tr, te = (data[k] for k in ['X', 'y', 'train', 'test'])
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('iqr', IQRClipper()), ('scaler', StandardScaler()), ('model', model)])
    baseline = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
    baseline_metrics = classification_metrics(baseline, X.iloc[te], y.iloc[te])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # Imputation, outlier clipping and scaling are refitted in every CV training fold.
    search = GridSearchCV(pipeline, {'model__' + k: v for k, v in grid.items()}, cv=cv,
        scoring='f1_weighted', n_jobs=JOBS, refit=True, error_score='raise')
    search.fit(X.iloc[tr], y.iloc[tr])
    best = search.best_estimator_
    tuned = classification_metrics(best, X.iloc[te], y.iloc[te])
    row = {'Algorithm': name, **tuned, 'CV_F1_mean': search.best_score_,
        'CV_F1_std': search.cv_results_['std_test_score'][search.best_index_],
        'Best_parameters': json.dumps(search.best_params_), 'Split_signature': data['signature']}
    display(pd.DataFrame([{'Stage': 'Baseline', **baseline_metrics}, {'Stage': 'Tuned', **tuned}]).round(5))
    changes = pd.DataFrame({'Metric': METRICS, 'Baseline': [baseline_metrics[m] for m in METRICS], 'Tuned': [tuned[m] for m in METRICS]})
    changes['Change'] = changes['Tuned'] - changes['Baseline']
    display(changes.round(6))
    print('Best parameters:', search.best_params_, '| CV weighted F1:', search.best_score_)
    key = slug(name)
    pd.DataFrame([row]).to_csv(RESULTS / f'{key}_summary.csv', index=False)
    changes.to_csv(RESULTS / f'{key}_tuning_changes.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(RESULTS / f'{key}_cv_search.csv', index=False)
    predictions = pd.DataFrame({'row_index': te, 'actual': y.iloc[te].to_numpy(), 'predicted': best.predict(X.iloc[te])})
    for i, cls in enumerate(best.classes_):
        predictions[f'probability_{cls}'] = best.predict_proba(X.iloc[te])[:, i]
    predictions.to_csv(RESULTS / f'{key}_predictions.csv', index=False)
    joblib.dump(best, MODELS / f'{key}.joblib')
    classification_plots(best, X.iloc[te], y.iloc[te], name)
    fitted = best.named_steps['model']
    if hasattr(fitted, 'feature_importances_'):
        importance = pd.Series(fitted.feature_importances_, index=X.columns).sort_values()
        importance.to_csv(RESULTS / f'{key}_feature_importance.csv', header=['Importance'])
        importance.plot.barh(figsize=(8, 5)); plt.title(name + ': feature importance'); plt.xlabel('Importance'); plt.ylabel('Feature')
        save_plot(key + '_importance.png')
    if name == 'Decision Tree':
        plt.figure(figsize=(20, 8)); plot_tree(fitted, feature_names=X.columns.tolist(), class_names=[str(c) for c in best.classes_], max_depth=3, filled=True, fontsize=8)
        plt.title('Decision Tree: first three levels'); save_plot('decision_tree_structure.png')
    if name == 'Logistic Regression':
        display(pd.DataFrame({'Feature': X.columns, 'Coefficient': fitted.coef_[0], 'Odds_ratio': np.exp(fitted.coef_[0])}))
    if name == 'MLP':
        plt.plot(fitted.loss_curve_); plt.xlabel('Iteration'); plt.ylabel('Training loss'); plt.title('Tuned MLP: training loss'); save_plot('mlp_loss.png')
    return row


def consolidate_classification(data):
    expected = list(classifier_specs())
    paths = [RESULTS / f'{slug(name)}_summary.csv' for name in expected]
    missing = [str(p.name) for p in paths if not p.exists()]
    if missing:
        raise ValueError(f'Incomplete comparison; run missing classifiers: {missing}')
    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if len(table) != 10 or set(table.Algorithm) != set(expected) or table.Algorithm.duplicated().any():
        raise ValueError('Comparison must contain exactly the ten required algorithms.')
    if table[METRICS].isna().any().any() or not np.isfinite(table[METRICS].to_numpy()).all():
        raise ValueError('Every mandatory metric must be finite.')
    if not table[METRICS].apply(lambda c: c.between(0, 1).all()).all():
        raise ValueError('Classification metrics must be between zero and one.')
    if not table.Split_signature.eq(data['signature']).all():
        raise ValueError('Dataset, features or held-out split differ; rerun all ten classifiers.')
    # Selection uses training CV only; test results are a final report, not a tuning objective.
    table = table.sort_values(['CV_F1_mean', 'Algorithm'], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, 'CV_rank', np.arange(1, 11))
    table.to_csv(RESULTS / 'classification_all_10_comparison.csv', index=False)
    display(table.drop(columns=['Split_signature', 'Best_parameters']).round(5))
    top2 = []
    for name in table.Algorithm.head(2):
        pipeline = joblib.load(MODELS / f'{slug(name)}.joblib')
        cv = cross_validate(clone(pipeline), data['X'].iloc[data['train']], data['y'].iloc[data['train']],
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring={'accuracy': 'accuracy', 'f1': 'f1_weighted'}, n_jobs=JOBS)
        for i in range(5):
            top2.append({'Algorithm': name, 'Fold': i + 1, 'Accuracy': cv['test_accuracy'][i], 'F1_weighted': cv['test_f1'][i]})
    pd.DataFrame(top2).to_csv(RESULTS / 'classification_top2_cv.csv', index=False)
    print('Selected using highest training CV weighted F1:', table.iloc[0].Algorithm)
    print('Top-two CV reuses tuning folds; it is not an unbiased nested-CV estimate. The untouched holdout is the final evaluation.')
    selected = table.iloc[0].Algorithm
    changes = pd.read_csv(RESULTS / f'{slug(selected)}_tuning_changes.csv')
    display(changes)
    print('Positive test metric changes:', changes.loc[changes.Change > 0, 'Metric'].tolist())
    print('If empty, the rubric improvement requirement remains unmet; do not claim improvement.')
    joblib.dump(joblib.load(MODELS / f'{slug(selected)}.joblib'), MODELS / 'final_classifier.joblib')
    return table


def marketing_data(show_eda=False):
    raw = pd.read_csv(data_path('marketing_campaign.csv'), sep=None, engine='python')
    if raw.shape[1] < 20:
        raise ValueError('Marketing file delimiter/schema could not be parsed.')
    audit_frame(raw)
    if show_eda:
        eda(raw, 'marketing')
    df = raw.drop(columns=['ID']).drop_duplicates().reset_index(drop=True)
    work = df.copy()
    dates = pd.to_datetime(work['Dt_Customer'], format='%d-%m-%Y', errors='raise')
    # Use the observed collection period, not the current calendar year.
    work['Age'] = dates.max().year - work['Year_Birth']
    work['Customer_Tenure_Days'] = (dates.max() - dates).dt.days
    spend = [c for c in work if c.startswith('Mnt')]
    purchases = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    work['Total_Spend'] = work[spend].sum(axis=1)
    work['Total_Purchases'] = work[purchases].sum(axis=1)
    # Implausible ages become missing; imputation is transparent and keeps row alignment.
    invalid_age = ~work.Age.between(18, 100)
    print('Implausible ages set missing:', int(invalid_age.sum()))
    work.loc[invalid_age, 'Age'] = np.nan
    excluded = ['Year_Birth', 'Dt_Customer', 'Response', 'Complain', 'Z_CostContact', 'Z_Revenue'] + [f'AcceptedCmp{i}' for i in range(1, 6)] + spend + purchases
    Xraw = work.drop(columns=excluded)
    constants = [c for c in Xraw if Xraw[c].nunique(dropna=True) <= 1]
    Xraw = Xraw.drop(columns=constants)
    numeric = Xraw.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in Xraw if c not in numeric]
    num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('iqr', IQRClipper()), ('scaler', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num', num, numeric), ('cat', cat, categorical)])
    X = prep.fit_transform(Xraw)
    print('Cleaned rows:', len(df), '| Clustering features:', Xraw.columns.tolist(), '| Matrix:', X.shape)
    # Clustering has no supervised holdout; preprocessing learns from the unlabeled cohort.
    # Resampling validation below refits this preprocessor on each subset independently.
    q1, q3 = Xraw[numeric].quantile(.25), Xraw[numeric].quantile(.75)
    display(((Xraw[numeric] < q1 - 1.5 * (q3-q1)) | (Xraw[numeric] > q3 + 1.5 * (q3-q1))).sum().to_frame('IQR outliers (clipped in pipeline)'))
    return {'raw': df, 'work': work, 'Xraw': Xraw, 'prep': prep, 'X': np.asarray(X, dtype=float)}


def cluster_scores(X, labels):
    if not 1 < len(np.unique(labels)) < len(labels):
        raise ValueError('Clustering metrics require 2 through n-1 clusters.')
    return dict(zip(CLUSTER_METRICS, [silhouette_score(X, labels), davies_bouldin_score(X, labels), calinski_harabasz_score(X, labels)]))


def cluster_plots(data, labels, name):
    key = slug(name)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(data['X'])
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels.astype(str), palette='colorblind', s=20, alpha=.65)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'); plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'); plt.title(name + ': PCA projection'); plt.legend(title='Cluster')
    save_plot(key + '_pca.png')
    print('PCA total explained variance:', pca.explained_variance_ratio_.sum())
    coords = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto', perplexity=min(30., (len(labels)-1)/3)).fit_transform(data['X'])
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels.astype(str), palette='colorblind', s=20, alpha=.65)
    plt.xlabel('t-SNE component 1'); plt.ylabel('t-SNE component 2'); plt.title(name + ': t-SNE projection'); plt.legend(title='Cluster')
    save_plot(key + '_tsne.png')
    assignments = data['work'].assign(Cluster=labels)
    assignments.to_csv(RESULTS / f'{key}_cluster_assignments.csv', index=False)
    profile = assignments.groupby('Cluster')[['Income', 'Age', 'Total_Spend', 'Total_Purchases', 'Recency']].mean()
    profile.insert(0, 'Customers', assignments.groupby('Cluster').size())
    display(profile.round(2))
    profile.to_csv(RESULTS / f'{key}_cluster_profiles.csv')
    # Campaign labels are accessed only here, after model fitting, for external interpretation.
    display(pd.crosstab(labels, data['raw']['Response'], normalize='index').rename_axis('Cluster'))


def resample_clusters(data, estimator, full_labels, name):
    rows = []
    for fold, (idx, _) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(data['Xraw']), 1):
        Xsub = clone(data['prep']).fit_transform(data['Xraw'].iloc[idx])
        labels = clone(estimator).fit_predict(Xsub)
        rows.append({'Algorithm': name, 'Fold': fold, **cluster_scores(Xsub, labels),
            'ARI_vs_full_fit': adjusted_rand_score(full_labels[idx], labels)})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / f'{slug(name)}_resampling.csv', index=False)
    display(frame.round(4))
    print('Unsupervised five-fold refit/stability check on 80% subsets, not held-out prediction accuracy. Hierarchical clustering has no predict method.')


def run_kmeans(data):
    X = data['X']; rows = []
    for k in range(1, min(11, len(X))):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        rows.append({'k': k, 'Inertia': model.inertia_})
    curve = pd.DataFrame(rows)
    # Geometric knee: largest perpendicular deviation below normalized endpoint chord.
    x = (curve.k - curve.k.min()) / (curve.k.max() - curve.k.min())
    y = (curve.Inertia - curve.Inertia.min()) / (curve.Inertia.max() - curve.Inertia.min())
    knee = int(curve.loc[((1 - x) - y).idxmax(), 'k'])
    knee = max(2, knee)
    plt.plot(curve.k, curve.Inertia, marker='o', label='Inertia')
    plt.axvline(knee, linestyle='--', color='#de8f05', label=f'Geometric knee: k={knee}')
    plt.xlabel('Number of clusters (k)'); plt.ylabel('Inertia'); plt.title('K-Means: elbow curve'); plt.legend()
    save_plot('kmeans_elbow.png')
    curve.to_csv(RESULTS / 'kmeans_elbow.csv', index=False)
    estimator = KMeans(n_clusters=knee, random_state=SEED, n_init=10)
    labels = estimator.fit_predict(X)
    row = {'Algorithm': 'K-Means', 'Clusters': knee, **cluster_scores(X, labels)}
    display(pd.DataFrame([row]).round(5))
    print('Cluster sizes:', np.bincount(labels).tolist(), '| k selected by geometric knee, independent of outcome labels.')
    pd.DataFrame([row]).to_csv(RESULTS / 'kmeans_metrics.csv', index=False)
    joblib.dump({'preprocessor': data['prep'], 'model': estimator, 'features': data['Xraw'].columns.tolist()}, MODELS / 'kmeans.joblib')
    cluster_plots(data, labels, 'K-Means')
    resample_clusters(data, estimator, labels, 'K-Means')
    return row


def run_agglomerative(data):
    X = data['X']
    path = RESULTS / 'kmeans_metrics.csv'
    if not path.exists():
        raise ValueError('Run K-Means first to establish the common elbow-derived k.')
    k = int(pd.read_csv(path).Clusters.iloc[0])
    # Full-data truncated dendrogram preserves all merges without thousands of leaf labels.
    rows, labels_by_method = [], {}
    for method in ['ward', 'complete', 'average']:
        Z = linkage(X, method=method)
        plt.figure(figsize=(12, 5))
        dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
        plt.xlabel('Merged groups (parentheses show observation counts)'); plt.ylabel('Merge distance')
        plt.title(f'Agglomerative: {method} linkage, full cohort, last 30 groups')
        save_plot('dendrogram_' + method + '.png')
        estimator = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = estimator.fit_predict(X)
        labels_by_method[method] = labels
        rows.append({'Linkage': method, 'Clusters': k, 'Minimum_cluster_size': int(np.bincount(labels).min()), **cluster_scores(X, labels)})
    comparison = pd.DataFrame(rows)
    display(comparison.round(5))
    comparison.to_csv(RESULTS / 'agglomerative_linkage_comparison.csv', index=False)
    # Reject trivial tiny-cluster solutions before using silhouette to select linkage.
    eligible = comparison[comparison.Minimum_cluster_size >= max(2, int(.02 * len(X)))]
    if eligible.empty:
        raise ValueError('All linkage solutions have tiny clusters; revisit features and k.')
    method = eligible.sort_values('Silhouette Score', ascending=False).iloc[0].Linkage
    labels = labels_by_method[method]
    estimator = AgglomerativeClustering(n_clusters=k, linkage=method)
    row = {'Algorithm': 'Agglomerative', 'Clusters': k, 'Linkage': method, **cluster_scores(X, labels)}
    print('Selection rule: minimum cluster size >=2% of cohort, then highest silhouette. Selected:', method)
    pd.DataFrame([row]).to_csv(RESULTS / 'agglomerative_metrics.csv', index=False)
    cluster_plots(data, labels, 'Agglomerative')
    resample_clusters(data, estimator, labels, 'Agglomerative')
    return row


def consolidate_clustering():
    table = pd.concat([pd.read_csv(RESULTS / (key + '_metrics.csv')) for key in ['kmeans', 'agglomerative']], ignore_index=True)
    if not np.isfinite(table[CLUSTER_METRICS].to_numpy()).all():
        raise ValueError('All mandatory clustering metrics must be finite.')
    display(table.round(5))
    table.to_csv(RESULTS / 'clustering_comparison.csv', index=False)
    return table
