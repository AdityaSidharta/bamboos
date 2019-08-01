from typing import Tuple

import numpy as np
import pandas as pd


def cv_split(
    X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Creates train and val dataset from the full dataset, given the corresponding train index and val index.
    This function is intended to be a accompanying function for Cross Validation from sklearn.model_selection.

    Examples:
        kf = KFold(n_split=3)
        for train_idx, val_idx in kf.split(X_train_val):
            X_train, X_val = cv_split(X_train_val, Y_train_val, train_idx, val_idx)

    Args:
        X (pd.DataFrame): Dataframe containing all of the full features
        y (pd.Series): Dataframe containing all of the full labels
        train_idx (np.ndarray): Array containing all training indices
        val_idx (np.ndarray): Array containing all validation indices

    Returns:
        X_train(pd.DataFrame): Dataframe containing all of the training features
        X_val(pd.DataFrame): Dataframe containing all of the validation features
    """
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx, :].reset_index(drop=True)
        X_val = X.iloc[val_idx, :].reset_index(drop=True)
    else:
        X_train = X[train_idx, :]
        X_val = X[val_idx, :]

    if isinstance(y, pd.Series):
        y_train = y[train_idx].reset_index(drop=True)
        y_val = y[val_idx].reset_index(drop=True)
    else:
        y_train = y[train_idx]
        y_val = y[val_idx]

    # TODO: Function returns a different data types (pandas or numpy) depending on X and y's datatypes.
    #  Do we want to fix return type to only pd.DataFrame/Series?
    return X_train, X_val, y_train, y_val
