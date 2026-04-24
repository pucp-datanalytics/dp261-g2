import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_validate

def train_baseline(pipe, X, y, name):
    """Entrena y persiste un baseline."""
    pipe.fit(X, y)
    joblib.dump(pipe, f'models/baseline_{name}.pkl')
    return pipe

def evaluate_model(pipe, X, y, cv, scoring):
    """Evalúa con cross-validation."""
    return cross_validate(
        pipe, X, y, cv=cv,
        scoring=scoring,
        return_train_score=True
    )

def log_experiment(name, params, metrics, path='models/experiments_log.csv'):
    """Registra un experimento en CSV."""
    row = {
        'model': name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        **params,
        **metrics
    }
    df = pd.DataFrame([row])
    # Si el archivo no existe, crea con header; si existe, agrega sin header
    import os
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=write_header, index=False)