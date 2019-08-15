from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from xgboost import DMatrix


def xgb_mape(preds: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    """
    Mean average precision error metric for evaluation in xgboost.

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Tuple of error name (str) and error (float)
    """
    labels = dtrain.get_label()
    mask = labels != 0
    return "mape", (np.fabs(labels - preds) / labels)[mask].mean()


def xgb_mape_exp(preds: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
    """
    Mean average precision error metric for evaluation in xgboost.
    NOTE: This will exponentiate the predictions first, in the case where our actual is logged

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Tuple of error name (str) and error (float)
    """
    labels = dtrain.get_label()
    mask = labels != 0
    return "mape_exp", (np.fabs(labels - np.exp(preds)) / labels)[mask].mean()


def xgb_pr_auc(preds: np.ndarray, lgb_train: DMatrix) -> Tuple[str, float]:
    """
    Precision Recall AUC (Area under Curve) of our prediction in lightgbxgboostm

    Args:
        preds: Array of predictions
        lgb_train: DMatrix of data

    Returns:
        Precision Recall AUC (Area under Curve)
    """
    labels = lgb_train.get_label()
    precision, recall, _ = precision_recall_curve(labels, preds)
    result = auc(recall, precision)
    return "pr_auc", result


def xgb_huber_approx(preds: np.ndarray, dtrain: DMatrix) -> Tuple[float, float]:
    """
    Huber loss (approximation) objective for xgboost.

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Gradient and hessian for huber loss
    """
    d = preds - dtrain.get_label()
    h = 1
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    # TODO returns np.arrays, not floats.
    return grad, hess


def xgb_fair(preds: np.ndarray, dtrain: DMatrix) -> Tuple[float, float]:
    """
    Fair loss objective for xgboost.

    y = c * abs(x) - c * np.log(abs(abs(x) + c))

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Gradient and hessian for fair loss
    """
    x = preds - dtrain.get_label()
    c = 1
    den = abs(x) + c
    grad = c * x / den
    hess = c * c / den ** 2
    return grad, hess


def xgb_log_cosh(preds: np.ndarray, dtrain: DMatrix) -> Tuple[float, float]:
    """
    Log-Cosh objective for xgboost.

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Gradient and hessian for log-cosh
    """
    x = preds - dtrain.get_label()
    grad = np.tanh(x)  # pylint: disable=assignment-from-no-return
    hess = 1 / np.cosh(x) ** 2
    return grad, hess


def xgb_mpse(preds: np.ndarray, dtrain: DMatrix) -> Tuple[float, float]:
    """
    Mean-Squared Percentage Error objective for xgboost

    Args:
        preds: Array of predictions
        dtrain: DMatrix of data

    Returns:
        Gradient and hessian for mean squared percentage error
    """
    yhat = dtrain.get_label()
    grad = 2.0 / yhat * (preds * 1.0 / yhat - 1)
    hess = 2.0 / (yhat ** 2)
    grad = np.where(np.isinf(grad), 0., grad)
    hess = np.where(np.isinf(hess), 0., hess)
    return grad, hess
