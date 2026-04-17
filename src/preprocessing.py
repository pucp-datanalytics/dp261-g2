import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def get_column_types(X: pd.DataFrame):
    """
    Detecta columnas numéricas y categóricas automáticamente.
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye el ColumnTransformer con pipelines para numéricas y categóricas.
    """

    num_cols, cat_cols = get_column_types(X)

    # Pipeline numérico
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline categórico
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor


def build_full_pipeline(X: pd.DataFrame, model):
    """
    Construye el pipeline completo (preprocesamiento + modelo).
    """
    preprocessor = build_preprocessor(X)

    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return full_pipeline