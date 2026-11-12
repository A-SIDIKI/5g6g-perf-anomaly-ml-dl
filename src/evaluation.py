"""Métriques d'évaluation communes à tous les modèles."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, precision_recall_curve)

def scores_regression(y_vrai, y_pred):
    """MAE, RMSE (dans l'unité de la cible) et R²."""
    return {
        "MAE": mean_absolute_error(y_vrai, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_vrai, y_pred)),
        "R2": r2_score(y_vrai, y_pred),
    }

def scores_classification(y_vrai, proba, seuil=0.5):
    """Précision, rappel et F1 au seuil donné ; AP (indépendant du seuil)."""
    y_pred = (proba >= seuil).astype(int)
    return {"precision": round(precision_score(y_vrai, y_pred, zero_division=0), 3),
            "rappel": round(recall_score(y_vrai, y_pred), 3),
            "f1": round(f1_score(y_vrai, y_pred), 3),
            "ap": round(average_precision_score(y_vrai, proba), 3)}


def seuil_f1(y_vrai, proba):
    """Seuil de décision qui maximise le F1."""
    precision, rappel, seuils = precision_recall_curve(y_vrai, proba)
    f1 = 2 * precision * rappel / (precision + rappel + 1e-12)
    return seuils[np.argmax(f1[:-1])]