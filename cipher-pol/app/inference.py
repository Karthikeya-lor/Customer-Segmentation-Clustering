"""Real model inference functions for the forthcoming web API; no UI/server started."""
from pathlib import Path
from functools import lru_cache
import sys
import math
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'r2'))
sys.path.insert(0, str(ROOT / 'r1'))
import review2_pipeline as r2
import review1_pipeline as r1


@lru_cache(maxsize=32)
def load_model(path_string):
    return joblib.load(ROOT / path_string)


def numeric_record(record, fields, positive=()):
    if not isinstance(record, dict):
        raise ValueError('Input must be one object with named fields.')
    values = {}
    for name in fields:
        if name not in record or isinstance(record[name], bool):
            raise ValueError(f'{name} is required and must be numeric.')
        try:
            number = float(record[name])
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be numeric.') from None
        if not math.isfinite(number):
            raise ValueError(f'{name} must be finite.')
        if (name in positive and number <= 0) or (name not in positive and number < 0):
            raise ValueError(f'{name} must be {"positive" if name in positive else "nonnegative"}.')
        values[name] = number
    return values


def predict_regression(record, model_name=None):
    choices = r1.regression_specs()
    if model_name is None:
        model_name = json.loads((ROOT / 'r1/results/regression_selection.json').read_text())['selected_model']
    if model_name not in choices:
        raise ValueError('Unknown regression model.')
    values = numeric_record(record, ['carat', 'depth', 'table', 'x', 'y', 'z'], positive=['carat', 'depth', 'table', 'x', 'y', 'z'])
    reference = pd.read_csv(ROOT / 'data/diamonds_sampled.csv')
    for name in ['cut', 'color', 'clarity']:
        if record.get(name) not in reference[name].unique():
            raise ValueError(f'Unsupported {name}; choose one of the dataset categories.')
        values[name] = record[name]
    X = r1.diamond_features(pd.DataFrame([values]))
    key = 'regression_' + r2.slug(model_name).replace('_regression', '')
    estimate = float(load_model(f'models/{key}.joblib').predict(X)[0])
    return {'model': model_name, 'predicted_price': estimate, 'price_unit': 'dataset price units; currency not verified'}


def predict_classification(record, model_name=None):
    if model_name is None:
        model_name = pd.read_csv(ROOT / 'r2/results/classification_all_10_comparison.csv').iloc[0].Algorithm
    if model_name not in r2.classifier_specs():
        raise ValueError('Unknown classification model.')
    fields = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Perimeter', 'Roundness', 'AspectRation']
    values = numeric_record(record, fields, positive=[field for field in fields if field != 'Eccentricity'])
    for name in ['Eccentricity', 'Extent', 'Roundness']:
        if values[name] > 1:
            raise ValueError(f'{name} must be at most 1.')
    values['Compactness'] = 4 * np.pi * values['Area'] / (values['Perimeter'] ** 2 + 1e-8)
    model = load_model(f'models/{r2.slug(model_name)}.joblib')
    X = pd.DataFrame([values])[list(model.feature_names_in_)]
    label = model.predict(X)[0]
    return {'model': model_name, 'predicted_class': label.item() if isinstance(label, np.generic) else label,
        'probabilities': {str(cls): float(value) for cls, value in zip(model.classes_, model.predict_proba(X)[0])}}


def assign_cluster(record):
    artifact = load_model('models/kmeans.joblib')
    categorical = ['Education', 'Marital_Status']
    numeric = [name for name in artifact['features'] if name not in categorical]
    values = numeric_record(record, numeric)
    if not 18 <= values['Age'] <= 100:
        raise ValueError('Age must be 18 through 100 at the dataset reference year.')
    for name in ['Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth', 'Customer_Tenure_Days', 'Total_Purchases']:
        if not values[name].is_integer():
            raise ValueError(f'{name} must be a whole number.')
    for name in categorical:
        fitted = artifact['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        choices = fitted.categories_[categorical.index(name)]
        if record.get(name) not in choices:
            raise ValueError(f'Unsupported {name}; choose a dataset category.')
        values[name] = record[name]
    X = pd.DataFrame([values])[artifact['features']]
    transformed = artifact['preprocessor'].transform(X)
    model = artifact['model']
    distances = model.transform(transformed)[0]
    return {'model': 'K-Means', 'cluster': int(model.predict(transformed)[0]),
        'centroid_distances': distances.tolist(), 'distance_unit': 'preprocessed feature distance',
        'note': 'Distances are not class probabilities. Agglomerative has no native new-record prediction.'}
