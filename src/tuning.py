# =========================================================
# Sprint 4 - Hyperparameter Tuning Module
# Rol 5: Experiment Tracker
# =========================================================

import os
import time
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV


def tune_model(
    pipe,
    param_grid,
    X,
    y,
    cv,
    scoring="recall",
    name="",
    models_dir="../models"
):
    """
    Tunea hiperparámetros, persiste el mejor modelo y retorna
    un registro compatible con experiments_log.csv.
    """

    start = time.time()

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X, y)

    elapsed = time.time() - start

    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, f"tuned_{name}.pkl")
    joblib.dump(grid.best_estimator_, path)

    result = {
        "model": name,
        "params": str(grid.best_params_),
        "date": datetime.today().strftime("%Y-%m-%d"),

        "cv_accuracy": None,
        "cv_precision": None,
        "cv_recall": grid.best_score_ if scoring == "recall" else None,
        "cv_f1": grid.best_score_ if scoring == "f1" else None,
        "cv_roc_auc": grid.best_score_ if scoring == "roc_auc" else None,

        "test_accuracy": None,
        "test_recall": None,
        "test_f1": None,
        "test_roc_auc": None,

        "recall_gap": None,
        "f1_gap": None,
        "train_time_s": round(elapsed, 2),

        "selected": False,
        "notes": f"Modelo tuneado con GridSearchCV usando scoring={scoring}",
        "tipo_modelo": "tuned",
        "dataset": "global",
        "path": path
    }

    return result, grid.best_estimator_, grid


def save_experiment(result, log_path="../models/experiments_log.csv"):
    """
    Agrega un experimento al archivo experiments_log.csv
    sin eliminar los registros anteriores.
    """

    df_new = pd.DataFrame([result])

    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)

        for col in df_new.columns:
            if col not in df_old.columns:
                df_old[col] = None

        for col in df_old.columns:
            if col not in df_new.columns:
                df_new[col] = None

        df_new = df_new[df_old.columns]
        df_final = pd.concat([df_old, df_new], ignore_index=True)

    else:
        df_final = df_new

    df_final.to_csv(log_path, index=False)

    print("✅ Experimento registrado correctamente")
    print(f"Archivo actualizado: {log_path}")
    print(f"Total registros: {len(df_final)}")

    return df_final


def validate_model_load(path, X_sample):
    """
    Valida que un modelo guardado pueda cargarse y predecir.
    """

    model = joblib.load(path)
    predictions = model.predict(X_sample)

    print("✅ Modelo cargado y validado correctamente")
    print("Predicciones:", predictions[:10])

    return predictions