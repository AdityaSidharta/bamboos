from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc


def rmse(y_actual: float, y_pred: float) -> float:
    """Root mean squared error

    Args:
        y_actual: Array of actual y values
        y_pred: Array of predicted y values

    Returns:
        Root mean squared error
    """
    return sqrt(mean_squared_error(y_actual, y_pred))


def mape(y_actual: float, y_pred: float) -> float:
    """Mean average precision error

    Args:
        y_actual: Array of actual y values
        y_pred: Array of predicted y values

    Returns:
        Mean average precision error
    """
    mask = y_actual != 0
    return (np.fabs(y_actual - y_pred) / y_actual)[mask].mean()


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray):
    """
    Area under Curve for Precision Recall Curve
    Args:
        y_true: Array of actual y values
        y_score: Array of predicted probability for all y values

    Returns:
        Area under Curve for Precision Recall Curve
    """
    assert y_true.shape == y_score.shape
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)
