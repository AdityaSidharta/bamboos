from typing import Any, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from bamboos.utils.cross_validation import cv_split


def get_oof_pred(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Out of Fold prediction for both training and testing dataset
    Args:
        model (Any): Any model class which have `fit` and `predict` method, and takes in numpy array as input
        X_train (pd.DataFrame): DataFrame containing features for the training dataset
        y_train (pd.Series): Series containing labels for the training dataset
        X_test (pd.DataFrame): DataFrame containing features for the test dataset
        n_fold (int): Number of Fold for Out of Fold prediction

    Returns:
        oof_train: Out-of-fold prediction for the training dataset
        oof_test_mean: Mean of the out-of-fold prediction for the test dataset
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    kf = KFold(n_fold)

    oof_train = np.zeros(shape=(n_train,))
    oof_test = np.zeros(shape=(n_fold, n_test))

    model = model()

    for idx, (dev_idx, val_idx) in enumerate(kf.split(X_train)):
        X_dev, X_val, y_dev, _ = cv_split(X_train, y_train, dev_idx, val_idx)
        model.fit(X_dev, y_dev)
        oof_train[val_idx] = model.predict(X_val)
        oof_test[idx, :] = model.predict(X_test)

    oof_test_mean = oof_test.mean(axis=0)
    return oof_train, oof_test_mean


def ensemble(
    model_list: List[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    ens_model: Any,
    n_fold: int = 5,
    return_detail: bool = False,
):
    """
    Performing Ensemble on our model
    Args:
        model_list: List of sklearn-like models that we would like to ensemble
        X_train: DataFrame containing features for the training dataset
        y_train: Series containing labels for the training dataset
        X_test: DataFrame containing features for the test dataset
        ens_model: Model that we use to ensemble model_list
        n_fold: Number of folds used in our Out of Fold prediction
        return_detail: True if detail about constructed ensemble model and dataset should be returned, False if
        only the final prediction should be returned

    Returns:
        prediction: Final prediction for the test dataset
        ens_model: Ensemble model which is fitted against the oof result of each model in model_list
        ensemble_train: Ensemble training dataset which is used to fit the ensemble model
        ensemble_test: Ensemble test dataset which is used to get the final prediction
    """
    n_model = len(model_list)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_ensemble_train = np.empty(shape=(n_train, n_model))
    X_ensemble_test = np.empty(shape=(n_test, n_model))
    column_name = []

    for idx, model in enumerate(model_list):
        model_name = model.__name__
        model_oof_train, model_oof_test = get_oof_pred(
            model, X_train, y_train, X_test, n_fold
        )
        X_ensemble_train[:, idx] = model_oof_train
        X_ensemble_test[:, idx] = model_oof_test
        column_name.append(model_name)

    ens_model.fit(X_ensemble_train, y_train)
    prediction = ens_model.predict(X_ensemble_test)

    if return_detail:
        ensemble_train = pd.DataFrame(X_ensemble_train, columns=column_name)
        ensemble_test = pd.DataFrame(X_ensemble_test, columns=column_name)
        return prediction, ens_model, ensemble_train, ensemble_test
    return prediction
