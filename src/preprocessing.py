import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import joblib


# =========================
# 1. Feature Builder
# =========================
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 👉 ejemplo simple (puedes agregar más lógica)
        if "total_nights" not in X.columns:
            if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(X.columns):
                X["total_nights"] = (
                    X["stays_in_weekend_nights"] +
                    X["stays_in_week_nights"]
                )

        return X


# =========================
# 2. Column groups
# =========================
def get_column_groups(df, target):
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return num_cols, cat_cols


# =========================
# 3. Preprocessor builder
# =========================
def build_preprocessor(num_cols, cat_cols):

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor


# =========================
# 4. Save artifact
# =========================
def save_artifact(obj, path):
    joblib.dump(obj, path)
