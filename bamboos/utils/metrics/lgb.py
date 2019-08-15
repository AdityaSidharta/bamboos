from typing import Tuple

import numpy as np
from lightgbm import Dataset
from sklearn.metrics import precision_recall_curve, auc


def lgb_mape(preds: np.ndarray, lgb_train: Dataset) -> Tuple[str, float, bool]:
    """
    Mean average precision error metric for evaluation in lightgbm.

    Args:
        preds: Array of predictions
        lgb_train: LightGBM Dataset

    Returns:
        Tuple of error name (str) and error (float)
    """
    labels = lgb_train.get_label()
    mask = labels != 0
    return "mape", (np.fabs(labels - preds) / labels)[mask].mean(), False


def lgb_mape_exp(preds: np.ndarray, lgb_train: Dataset) -> Tuple[str, float, bool]:
    """
    Mean average precision error metric for evaluation in lightgbm.
    NOTE: This will exponentiate the predictions first, in the case where our actual is logged

    Args:
        preds: Array of predictions
        lgb_train: LightGBM Dataset

    Returns:
        Tuple of error name (str) and error (float)
    """
    labels = lgb_train.get_label()
    mask = labels != 0
    return "mape_exp", (np.fabs(labels - np.exp(preds)) / labels)[mask].mean(), False


def lgb_pr_auc(preds: np.ndarray, lgb_train: Dataset) -> Tuple[str, float, bool]:
    """
    Precision Recall AUC (Area under Curve) of our prediction in lightgbm

    Args:
        preds: Array of predictions
        lgb_train: LightGBM Dataset

    Returns:
        Precision Recall AUC (Area under Curve)
    """
    labels = lgb_train.get_label()
    precision, recall, _ = precision_recall_curve(labels, preds)
    return "pr_auc", auc(recall, precision), True
