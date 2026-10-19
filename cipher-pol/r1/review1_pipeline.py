"""Review 1 regression and Part A workflows, shared by individual and track notebooks."""
from pathlib import Path
import sys
import json
import hashlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'r2'))
import review2_pipeline as r2

RESULTS = ROOT / 'r1/results'
RESULTS.mkdir(exist_ok=True)
JOBS = int(os.environ.get('REVIEW2_JOBS', '1'))
METRICS = ['R2', 'RMSE', 'MAE']
PART_A = list(r2.classifier_specs())[:5]


def save_plot(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=140, bbox_inches='tight')
    plt.show()
    plt.close('all')


def diamond_features(frame):
    """Deterministic feature preparation, also usable for future website requests."""
    X = frame.drop(columns=['id: 0', 'price'], errors='ignore').copy()
    for col in ['x', 'y', 'z']:
        X.loc[X[col] <= 0, col] = np.nan
    X['volume'] = X.x * X.y * X.z
    return X


def diamonds_data():
    path = ROOT / 'data/diamonds_sampled.csv'
    raw = pd.read_csv(path)
    r2.audit_frame(raw)
    display(raw.price.describe().to_frame('price'))
    invalid = {c: int((raw[c] <= 0).sum()) for c in ['x', 'y', 'z']}
    print('Nonpositive dimensions converted to missing:', invalid)
    df = raw.drop(columns=['id: 0']).dropna(subset=['price']).drop_duplicates().reset_index(drop=True)
    X, y = diamond_features(df), df.price
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Diamond targets must be finite nonnegative prices.')
    bins = pd.qcut(y, q=10, duplicates='drop')
    tr, te = train_test_split(np.arange(len(df)), test_size=.2, random_state=42, stratify=bins)
    manifest = {'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'features': X.columns.tolist(),
        'preprocessing': 'median-IQR1.5-scale-onehot-dropfirst;volume;v1', 'seed': 42,
        'stratification': 'price deciles', 'train_indices': tr.tolist(), 'test_indices': te.tolist()}
    manifest['signature'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (RESULTS / 'regression_split.json').write_text(json.dumps(manifest, indent=2))
    print('Cleaned rows:', len(df), '| Train:', len(tr), '| Test:', len(te))
    numeric = X.select_dtypes(include=np.number).columns
    q1, q3 = X[numeric].quantile(.25), X[numeric].quantile(.75)
    outliers = ((X[numeric] < q1-1.5*(q3-q1)) | (X[numeric] > q3+1.5*(q3-q1))).sum().to_frame('IQR_outliers')
    display(outliers); outliers.to_csv(RESULTS / 'diamond_outlier_audit.csv')
    return {'X': X, 'y': y, 'train': tr, 'test': te, 'signature': manifest['signature']}


def diamond_eda(data, section):
    df = data['X'].assign(price=data['y'])
    if section == 'distributions':
        columns = data['X'].select_dtypes(include=np.number).columns
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, col in zip(axes.flat, columns):
            ax.hist(df[col].dropna(), bins=30, color='#0173b2'); ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Rows')
        for ax in list(axes.flat)[len(columns):]:
            ax.remove()
        fig.suptitle('Diamond numeric feature distributions')
        save_plot('diamond_distributions.png')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(axes, ['cut', 'color', 'clarity']):
            df[col].value_counts().plot.bar(ax=ax, rot=30); ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Rows')
        fig.suptitle('Diamond quality category distributions'); save_plot('diamond_categories.png')
    elif section == 'correlation':
        plt.figure(figsize=(11, 8)); sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='vlag', vmin=-1, vmax=1, annot=True, fmt='.2f')
        plt.title('Diamond feature and price correlations'); plt.xlabel('Feature'); plt.ylabel('Feature'); save_plot('diamond_correlation.png')
    elif section == 'target':
        plt.hist(df.price, bins=30); plt.xlabel('Price (dataset units)'); plt.ylabel('Rows'); plt.title('Diamond price distribution'); save_plot('diamond_target.png')
    elif section == 'scatter':
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, col in zip(axes, ['carat', 'x']):
            ax.scatter(df[col], df.price, alpha=.35, s=12); ax.set_title(f'{col} versus price'); ax.set_xlabel(col); ax.set_ylabel('Price (dataset units)')
        save_plot('diamond_feature_target.png')


def regression_specs():
    return {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge Regression': (Ridge(), {'alpha': [.1, 1., 10.]}),
        'Lasso Regression': (Lasso(alpha=1., max_iter=100000, tol=.001), {'alpha': [.1, 1., 10.]}),
        'ElasticNet Regression': (ElasticNet(alpha=.01, max_iter=100000, tol=.001, random_state=42), {'alpha': [.001, .01, .1], 'l1_ratio': [.1, .5, .9]}),
        'Polynomial Regression': (LinearRegression(), {'__degree': [2, 3]}),
        'Decision Tree Regression': (DecisionTreeRegressor(random_state=42), {'max_depth': [3, 6, 12, None], 'min_samples_leaf': [1, 3]}),
        'Random Forest Regression': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1), {'n_estimators': [100, 200], 'max_depth': [None, 12], 'min_samples_leaf': [1, 3]}),
        'Gradient Boosting Regression': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 150], 'learning_rate': [.05, .1, .15], 'max_depth': [2, 3]}),
        'Support Vector Regression': (TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()), {'regressor__C': [1, 10, 100], 'regressor__kernel': ['linear', 'rbf']}),
        # Explicit tree search avoids brute-force squared-norm cancellation at
        # zero/near-zero distances and makes repeated CV fits reproducible.
        'KNN Regression': (KNeighborsRegressor(algorithm='kd_tree'), {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']})
    }


def regression_pipeline(name, data):
    model, parameters = regression_specs()[name]
    numeric = data['X'].select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in data['X'] if c not in numeric]
    steps = [('imputer', SimpleImputer(strategy='median')), ('iqr', r2.IQRClipper()), ('scaler', StandardScaler())]
    if name == 'Polynomial Regression':
        # Expand continuous predictors only; avoid redundant one-hot squares/interactions.
        steps += [('polynomial', PolynomialFeatures(degree=2, include_bias=False))]
    prep = ColumnTransformer([('num', Pipeline(steps), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), categorical)])
    pipeline = Pipeline([('preprocessor', prep), ('model', clone(model))])
    grid = {('preprocessor__num__polynomial__degree' if key == '__degree' else 'model__' + key): value for key, value in parameters.items()}
    return pipeline, grid


def scores(y, pred):
    return {'R2': r2_score(y, pred), 'RMSE': root_mean_squared_error(y, pred), 'MAE': mean_absolute_error(y, pred)}


def regression_diagnostics(y, pred, name, key):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(pred, y-pred, alpha=.4, s=14); axes[0].axhline(0, color='#de8f05', linestyle='--', label='Zero residual')
    axes[0].set_title(name + ': residuals'); axes[0].set_xlabel('Predicted price'); axes[0].set_ylabel('Actual minus predicted'); axes[0].legend()
    axes[1].scatter(y, pred, alpha=.4, s=14)
    lim = [min(np.min(y), np.min(pred)), max(np.max(y), np.max(pred))]
    axes[1].plot(lim, lim, '--', color='#de8f05', label='Perfect prediction'); axes[1].legend()
    axes[1].set_title(name + ': predicted versus actual'); axes[1].set_xlabel('Actual price'); axes[1].set_ylabel('Predicted price')
    save_plot(key + '_diagnostics.png')


def fitted_feature_names(pipeline):
    prep = pipeline.named_steps['preprocessor']
    num_names = list(prep.transformers_[0][2])
    num = prep.named_transformers_['num']
    if 'polynomial' in num.named_steps:
        num_names = num.named_steps['polynomial'].get_feature_names_out(num_names).tolist()
    cat_names = prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(prep.transformers_[1][2]).tolist()
    return num_names + cat_names


def run_regressor(name, data):
    X, y, tr, te = (data[k] for k in ['X', 'y', 'train', 'test'])
    pipeline, grid = regression_pipeline(name, data)
    baseline = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
    baseline_scores = scores(y.iloc[te], baseline.predict(X.iloc[te]))
    cv = KFold(5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, grid or [{}], cv=cv, scoring='r2', n_jobs=JOBS, error_score='raise', refit=True)
    search.fit(X.iloc[tr], y.iloc[tr])
    best = search.best_estimator_
    pred = best.predict(X.iloc[te]); tuned = scores(y.iloc[te], pred)
    row = {'Model': name, **tuned, 'CV_R2_Mean': search.best_score_, 'CV_R2_Std': search.cv_results_['std_test_score'][search.best_index_],
        'Best_parameters': json.dumps(search.best_params_), 'Split_signature': data['signature']}
    changes = pd.DataFrame({'Metric': METRICS, 'Baseline': [baseline_scores[m] for m in METRICS], 'Tuned': [tuned[m] for m in METRICS]})
    changes['Change'] = changes.Tuned - changes.Baseline
    print(name, '| Best parameters:', search.best_params_, '| Training CV R2:', search.best_score_)
    display(changes.round(6))
    key = 'regression_' + r2.slug(name).replace('_regression', '')
    pd.DataFrame([row]).to_csv(RESULTS / f'{key}_summary.csv', index=False)
    changes.to_csv(RESULTS / f'{key}_tuning.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(RESULTS / f'{key}_cv_search.csv', index=False)
    pd.DataFrame({'row_index': te, 'actual': y.iloc[te].to_numpy(), 'baseline_predicted': baseline.predict(X.iloc[te]), 'predicted': pred, 'residual': y.iloc[te].to_numpy()-pred}).to_csv(RESULTS / f'{key}_predictions.csv', index=False)
    joblib.dump(best, ROOT / 'models' / f'{key}.joblib')
    regression_diagnostics(y.iloc[te].to_numpy(), pred, name.replace(' Regression', ''), key)
    model = best.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=fitted_feature_names(best)).sort_values()
        importance.to_csv(RESULTS / f'{key}_importance.csv', header=['Importance'])
        importance.tail(15).plot.barh(figsize=(9, 5)); plt.title(name + ': feature importance'); plt.xlabel('Importance'); plt.ylabel('Encoded feature'); save_plot(key + '_importance.png')
    if hasattr(model, 'coef_'):
        coefficients = pd.DataFrame({'Feature': fitted_feature_names(best), 'Coefficient': model.coef_})
        display(coefficients)
        coefficients.to_csv(RESULTS / f'{key}_coefficients.csv', index=False)
        if name == 'Lasso Regression':
            print('Lasso zero coefficients:', int(np.isclose(model.coef_, 0).sum()), '/', len(model.coef_))
    return row


def consolidate_regression(data):
    names = list(regression_specs())
    paths = [RESULTS / ('regression_' + r2.slug(n).replace('_regression', '') + '_summary.csv') for n in names]
    if any(not p.exists() for p in paths):
        raise ValueError('Run all ten regressors before consolidation.')
    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if len(table) != 10 or set(table.Model) != set(names) or table.Model.duplicated().any():
        raise ValueError('Exactly ten required regression models are mandatory.')
    if not table.Split_signature.eq(data['signature']).all() or not np.isfinite(table[METRICS].to_numpy()).all():
        raise ValueError('Results must be finite and from the same dataset/split/preprocessing.')
    if (table[['RMSE', 'MAE']] < 0).any().any():
        raise ValueError('Regression errors cannot be negative.')
    table = table.sort_values('R2', ascending=False).reset_index(drop=True)
    table.insert(0, 'Rank_by_test_R2', np.arange(1, 11))
    table.to_csv(RESULTS / 'regression_comparison.csv', index=False)
    display(table.drop(columns=['Split_signature', 'Best_parameters']).round(6))
    # Report CV for the two top test-ranked algorithms as explicitly requested, and
    # separately select the deployment artifact by training CV to avoid test-led selection.
    leading = table.head(2).Model.tolist()
    selected = table.sort_values('CV_R2_Mean', ascending=False).iloc[0].Model
    cv_rows = []
    for name in dict.fromkeys(leading + [selected]):
        key = 'regression_' + r2.slug(name).replace('_regression', '')
        fitted = joblib.load(ROOT / 'models' / f'{key}.joblib')
        values = cross_val_score(clone(fitted), data['X'].iloc[data['train']], data['y'].iloc[data['train']], cv=KFold(5, shuffle=True, random_state=42), scoring='r2', n_jobs=JOBS)
        cv_rows += [{'Model': name, 'Fold': i+1, 'R2': value, 'Top_two_test_rank': name in leading} for i, value in enumerate(values)]
    display(pd.DataFrame(cv_rows).round(6)); pd.DataFrame(cv_rows).to_csv(RESULTS / 'regression_top2_cv.csv', index=False)
    key = 'regression_' + r2.slug(selected).replace('_regression', '')
    joblib.dump(joblib.load(ROOT / 'models' / f'{key}.joblib'), ROOT / 'models/final_regressor.joblib')
    (RESULTS / 'regression_selection.json').write_text(json.dumps({'selected_model': selected, 'selection_rule': 'highest training five-fold CV R2', 'best_test_R2_model': table.iloc[0].Model, 'split_signature': data['signature']}, indent=2))
    print('CV-selected final model:', selected, '| Highest test R2:', table.iloc[0].Model)
    print('CV reuses search folds and is not unbiased nested CV. All test metrics are reported in original price units.')
    final = joblib.load(ROOT / 'models/final_regressor.joblib')
    regression_diagnostics(data['y'].iloc[data['test']].to_numpy(), final.predict(data['X'].iloc[data['test']]), 'CV-selected final model', 'regression_final')
    # Also provide mandatory diagnostic plots for the highest held-out R2 model.
    name = table.iloc[0].Model
    key = 'regression_' + r2.slug(name).replace('_regression', '')
    best_test = joblib.load(ROOT / 'models' / f'{key}.joblib')
    regression_diagnostics(data['y'].iloc[data['test']].to_numpy(), best_test.predict(data['X'].iloc[data['test']]), 'Highest test R2 model', 'regression_best_test')
    return table


def run_part_a(name, data):
    return r2.run_classifier(name, data)


def consolidate_part_a(data):
    table = pd.concat([pd.read_csv(r2.RESULTS / f'{r2.slug(name)}_summary.csv') for name in PART_A], ignore_index=True)
    if set(table.Algorithm) != set(PART_A) or len(table) != 5 or not table.Split_signature.eq(data['signature']).all() or not np.isfinite(table[r2.METRICS]).all().all():
        raise ValueError('Five complete Part A rows with the current shared split are required.')
    table = table.sort_values('F1_weighted', ascending=False).reset_index(drop=True)
    display(table.drop(columns=['Best_parameters', 'Split_signature']).round(6))
    table.to_csv(RESULTS / 'classification_part_a_comparison.csv', index=False)
    return table
