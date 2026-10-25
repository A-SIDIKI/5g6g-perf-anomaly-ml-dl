"""Métriques d'évaluation communes à tous les modèles."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def scores_regression(y_vrai, y_pred):
    """MAE, RMSE (dans l'unité de la cible) et R²."""
    return {
        "MAE": mean_absolute_error(y_vrai, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_vrai, y_pred)),
        "R2": r2_score(y_vrai, y_pred),
    }