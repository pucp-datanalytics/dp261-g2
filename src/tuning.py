# Sprint 4: Optimización de hiperparámetros
# Módulo de tuning para los modelos seleccionados en el Sprint 3.
#
# Modelos candidatos (ordenados por CV Recall en Sprint 3):
#   1. baseline_dt.pkl  — Decision Tree    (CV Recall: 0.761, Rank 1)
#   2. baseline_rf.pkl  — Random Forest    (CV Recall: 0.752, Rank 2, mejor AUC: 0.927)
#   3. baseline_xgb.pkl — XGBoost          (CV Recall: 0.714, Rank 3, gap mínimo)
#
# Métrica principal: Recall (detectar cancelaciones hoteleras)
# Métricas secundarias: F1, AUC-ROC


def tune_model(pipe, param_grid, X, y, cv):
    """Optimiza hiperparámetros de un pipeline usando GridSearchCV o RandomizedSearchCV."""
    pass  # TODO Sprint 4


def tune_decision_tree(preproc, X, y, cv):
    """Tuning de Decision Tree: max_depth, min_samples_split, min_samples_leaf."""
    pass  # TODO Sprint 4


def tune_random_forest(preproc, X, y, cv):
    """Tuning de Random Forest: n_estimators, max_depth, max_features."""
    pass  # TODO Sprint 4


def tune_xgboost(preproc, X, y, cv):
    """Tuning de XGBoost: n_estimators, max_depth, learning_rate, subsample."""
    pass  # TODO Sprint 4
