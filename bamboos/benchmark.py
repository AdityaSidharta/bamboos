import os
from collections import OrderedDict
from typing import Any, List, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

from bamboos.utils.cross_validation import cv_split
from bamboos.utils.metrics.sklearn import pr_auc_score
from bamboos.utils.model.model_zoo import (
    binary_model_dict,
    multiclass_model_dict,
    regression_model_dict,
)
from bamboos.utils.vis import barplot, boxplot


def get_default_metric(model_type: str):
    """
    Get default metrics, metrics_proba, metrics_kwargs, sort_by, ascending for all the supported target types
    Args:
        model_type: One of the following : ['regression', 'binary', 'multiclass']

    Returns:
        Default metrics, metrics_proba, metrics_kwargs, sort_by, ascending
    """
    if model_type == "regression":
        metrics = [
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            explained_variance_score,
        ]
        metrics_proba = []  # type: List[Any]
        metrics_kwargs = {}  # type: dict
        sort_by = mean_absolute_error.__name__
        ascending = True
    elif model_type == "binary":
        metrics = [accuracy_score, recall_score, precision_score, f1_score]
        metrics_proba = [roc_auc_score, pr_auc_score]
        metrics_kwargs = {}
        sort_by = f1_score.__name__
        ascending = False
    elif model_type == "multiclass":
        metrics = [accuracy_score, recall_score, precision_score, f1_score]
        metrics_proba = [roc_auc_score]
        metrics_kwargs = {
            "recall_score": {"average": "macro"},
            "precision_score": {"average": "macro"},
            "f1_score": {"average": "macro"},
            "roc_auc_score": {"average": "macro"},
        }
        sort_by = f1_score.__name__
        ascending = False
    else:
        raise ValueError("model_type not in [regression, binary, multiclass]")
    return metrics, metrics_proba, metrics_kwargs, sort_by, ascending


def bm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metrics: List[Any],
    metrics_proba: List[Any],
    metrics_kwargs: dict,
    model_dict: dict,
):
    """
    Perform benchmark prediction on the given validation dataset for all model supplied under model_dictionary
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        X_val: Array of features, used to validate the model
        y_val: Array of label, used to validate the model
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        model_dict: Model_dictionary, containing the model_name as the key and catalyst.ml.model object as value.

    Returns:
        DataFrame, which contains all of the metrics value for each of the model specified under model_dictionary.
    """
    result_row = []
    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        result_dict: dict = OrderedDict()
        result_dict["model_name"] = model_name
        metrics = [] if metrics is None else metrics
        metrics_proba = [] if metrics_proba is None else metrics_proba
        metrics_kwargs = {} if metrics_kwargs is None else metrics_kwargs
        for metric in metrics:
            if metric.__name__ in metrics_kwargs.keys():
                result_dict[metric.__name__] = model.evaluate(
                    X_val, y_val, metric, **metrics_kwargs[metric.__name__]
                )
            else:
                result_dict[metric.__name__] = model.evaluate(X_val, y_val, metric)
        for metric_proba in metrics_proba:
            if metric_proba.__name__ in metrics_kwargs.keys():
                result_dict[metric_proba.__name__] = model.evaluate_proba(
                    X_val, y_val, metric_proba, **metrics_kwargs[metric_proba.__name__]
                )
            else:
                result_dict[metric_proba.__name__] = model.evaluate_proba(
                    X_val, y_val, metric_proba
                )
        result_row.append(result_dict)
    result_df = pd.DataFrame(result_row)
    return result_df


def bm_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    metrics: List[Any],
    metrics_proba: List[Any],
    metric_kwargs: dict,
    model_dict: dict,
):
    """
    Perform cross validation benchmark with all models specified under model_dictionary, using the metrics defined.
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        cv: Number of cross-validation fold
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        model_dict: Model_dictionary, containing the model_name as the key and catalyst.ml.model object as value.

    Returns:
        DataFrame, which contains all of the metrics value for each of the model specified under model_dictionary,
        as well as the cross-validation index.
    """
    result_cv_df = pd.DataFrame()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    for cv_idx, (dev_idx, val_idx) in enumerate(kf.split(X_train)):
        X_dev, X_val, y_dev, y_val = cv_split(X_train, y_train, dev_idx, val_idx)
        df = bm(
            X_dev,
            y_dev,
            X_val,
            y_val,
            metrics,
            metrics_proba,
            metric_kwargs,
            model_dict,
        )
        df["cv_idx"] = cv_idx
        result_cv_df = pd.concat([result_cv_df, df])
    return result_cv_df


def aggregate(
    result_cv_df: pd.DataFrame,
    metrics: Optional[List[Any]],
    metrics_proba: Optional[List[Any]],
):
    """
    Perform Aggregation of values from different cross_validation across all metrics, grouped by the model_name
    Args:
        result_cv_df: DataFrame, containing values for different metrics and model
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class

    Returns:
        Aggregated DataFrame values with different metrics as columns, model_name as rows.
    """
    metrics = [] if metrics is None else metrics
    metrics_proba = [] if metrics_proba is None else metrics_proba
    metrics_cols = [metric.__name__ for metric in metrics] + [
        metric_proba.__name__ for metric_proba in metrics_proba
    ]
    aggregate_df = (
        result_cv_df.groupby("model_name")[metrics_cols].agg("mean").reset_index()
    )
    return aggregate_df


def sort(result_df: pd.DataFrame, sort_by: Optional[str], ascending: bool):
    """
    Sort the given dataframe given the sort column_name, and the sorting_procedure
    Args:
        result_df: DataFrame, containing the column specified in sort_by
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        ascending: Specify the ordering in the sort

    Returns:
        Sorted DataFrame
    """
    if sort_by:
        result_df = result_df.sort_values(by=sort_by, ascending=ascending)
    return result_df


def plot_save(
    result_df: pd.DataFrame,
    metrics: Optional[list],
    metrics_proba: Optional[list],
    plot: bool,
    folder_path: Optional[str],
):
    """
    Plot and save the plot of the DataFrame, if needed
    Args:
        result_df: DataFrame to be plotted
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:

    """
    metrics = [] if metrics is None else metrics
    metrics_proba = [] if metrics_proba is None else metrics_proba
    metrics_cols = [metric.__name__ for metric in metrics] + [
        metric_proba.__name__ for metric_proba in metrics_proba
    ]
    if plot:
        for metric in metrics_cols:
            barplot(result_df, metric, "model_name")
    if folder_path:
        for metric in metrics_cols:
            barplot(
                result_df,
                metric,
                "model_name",
                filename=os.path.join(folder_path, "{}_plot.png".format(metric)),
            )


def plot_save_cv(result_df: pd.DataFrame, metrics, metrics_proba, plot, folder_path):
    """
    Plot and save the plot of the DataFrame, if needed
    Args:
        result_df: DataFrame to be plotted
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:

    """
    metrics_cols = [metric.__name__ for metric in metrics] + [
        metric_proba.__name__ for metric_proba in metrics_proba
    ]
    if plot:
        for metric in metrics_cols:
            boxplot(result_df, metric, "model_name")
    if folder_path:
        for metric in metrics_cols:
            boxplot(
                result_df,
                metric,
                "model_name",
                filename=os.path.join(folder_path, "{}_cv_plot.png".format(metric)),
            )


def regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metrics: List[Any] = None,
    metrics_kwargs: dict = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform benchmark for regression problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        X_val: Array of features, used to validate the model
        y_val: Array of label, used to validate the model
        metrics: List of metrics that we will use to score our validation performance
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for regression benchmark

    Examples:
        >>> X_full, y_full = sklearn.datasets.load_boston(True)
        >>> X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_full, y_full, test_size = 0.20)
        >>> regression(X_train, y_train, X_val, y_val)
    """
    if metrics is None:
        # metrics_proba will be empty for regression
        metrics, _, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "regression"
        )
    result_df = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        [],
        metrics_kwargs,  # type: ignore
        regression_model_dict(),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save(result_df, metrics, [], plot, folder_path)
    return result_df


def binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metrics: List[Any] = None,
    metrics_proba: List[Any] = None,
    metrics_kwargs: Optional[dict] = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform benchmark for classification problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        X_val: Array of features, used to validate the model
        y_val: Array of label, used to validate the model
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for classification benchmark

    Examples:
        >>> X_full, y_full = sklearn.datasets.load_breast_cancer(True)
        >>> X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_full, y_full, test_size = 0.20)
        >>> binary(X_train, y_train, X_val, y_val)
    """
    if metrics is None:
        metrics, metrics_proba, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "binary"
        )
    result_df = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        metrics_proba,
        metrics_kwargs,  # type: ignore
        binary_model_dict(),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save(result_df, metrics, metrics_proba, plot, folder_path)
    return result_df


def multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_class: int,
    metrics: Optional[list] = None,
    metrics_proba: Optional[list] = None,
    metrics_kwargs: Optional[dict] = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform benchmark for multi-classification problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        X_val: Array of features, used to validate the model
        y_val: Array of label, used to validate the model
        num_class: number of class in the labels. All of the labels must be within [0, num_class - 1]
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for classification benchmark

    Examples:
        >>> X_full, y_full = sklearn.datasets.load_digits(10, True)
        >>> X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_full, y_full, test_size = 0.20)
        >>> multiclass(X_train, y_train, X_val, y_val, num_class = 10)
    """
    if metrics is None:
        metrics, metrics_proba, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "multiclass"
        )
    result_df = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        metrics_proba,
        metrics_kwargs,  # type: ignore
        multiclass_model_dict(num_class=num_class),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save(result_df, metrics, metrics_proba, plot, folder_path)
    return result_df


def regression_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    metrics: List[Any] = None,
    metrics_kwargs: dict = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform cross-validation benchmark for regression problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        cv: Number of cross validation to be performed
        metrics: List of metrics that we will use to score our validation performance
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for cross-validated regression benchmark


    Examples:
        >>> X_full, Y_full = sklearn.datasets.load_boston(True)
        >>> regression_cv(X_full, Y_full)
    """
    if metrics is None:
        metrics, metrics_proba, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "regression"
        )
    result_df = bm_cv(
        X_train,
        y_train,
        cv,
        metrics,
        metrics_proba,
        metrics_kwargs,  # type: ignore
        regression_model_dict(),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save_cv(result_df, metrics, metrics_proba, plot, folder_path)
    aggregate_df = aggregate(result_df, metrics, metrics_proba)
    aggregate_df = sort(aggregate_df, sort_by, is_smaller_better)
    return aggregate_df


def binary_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    metrics: List[Any] = None,
    metrics_proba: List[Any] = None,
    metrics_kwargs: dict = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform cross-validation benchmark for binary problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        cv: Number of cross validation to be performed
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for cross-validated binary benchmark

    Examples:
        >>> X_full, Y_full = sklearn.datasets.load_breast_cancer(True)
        >>> binary_cv(X_full, Y_full)
    """
    if metrics is None:
        metrics, metrics_proba, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "binary"
        )
    result_df = bm_cv(
        X_train,
        y_train,
        cv,
        metrics,
        metrics_proba,
        metrics_kwargs,  # type: ignore
        binary_model_dict(),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save_cv(result_df, metrics, metrics_proba, plot, folder_path)
    aggregate_df = aggregate(result_df, metrics, metrics_proba)
    aggregate_df = sort(aggregate_df, sort_by, is_smaller_better)
    return aggregate_df


def multiclass_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_class: int,
    cv: int = 5,
    metrics: List[Any] = None,
    metrics_proba: List[Any] = None,
    metrics_kwargs: dict = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    """
    Perform cross-validation benchmark for multiclass problem
    Args:
        X_train: Array of features, used to train the model
        y_train: Array of label, used to train the model
        num_class: number of class in the labels. All of the labels must be within [0, num_class - 1]
        cv: Number of cross validation to be performed
        metrics: List of metrics that we will use to score our validation performance
        metrics_proba : List of metrics that we will use to score our validation performance.
        This is only applicable for classification problem. The metrics under `metrics_proba` uses the predicted
        probability instead of predicted class
        metrics_kwargs: Dictionary containing the extra arguments needed for specific metrics,
         listed in metrics and metrics_proba
        sort_by: Column which will be used to sort the dataframe. None if the dataframe should not be sorted
        is_smaller_better: Specify the ordering in the sort
        plot: True if plot should be shown, False otherwise
        folder_path: String containing the folder_path where all of the plots will be stored, None otherwise

    Returns:
        DataFrame containing the result for cross-validated multiclass benchmark

    Examples:
        >>> X_full, Y_full = sklearn.datasets.load_digits(10, True)
        >>> multiclass_cv(X_full, Y_full, num_class=10)
    """
    if metrics is None:
        metrics, metrics_proba, metrics_kwargs, sort_by, is_smaller_better = get_default_metric(
            "multiclass"
        )
    result_df = bm_cv(
        X_train,
        y_train,
        cv,
        metrics,
        metrics_proba,
        metrics_kwargs,  # type: ignore
        multiclass_model_dict(num_class=num_class),
    )
    result_df = sort(result_df, sort_by, is_smaller_better)
    plot_save_cv(result_df, metrics, metrics_proba, plot, folder_path)
    aggregate_df = aggregate(result_df, metrics, metrics_proba)
    aggregate_df = sort(aggregate_df, sort_by, is_smaller_better)
    return aggregate_df


def benchmark(
    kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_class: int,
    metrics: Optional[list] = None,
    metrics_proba: Optional[list] = None,
    metrics_kwargs: Optional[dict] = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    if kind in ["reg", "regression"]:
        result_df = regression(
            X_train,
            y_train,
            X_val,
            y_val,
            metrics,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    elif kind in ["bin", "binary"]:
        result_df = binary(
            X_train,
            y_train,
            X_val,
            y_val,
            metrics,
            metrics_proba,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    elif kind in ["multi", "multiclass"]:
        result_df = multiclass(
            X_train,
            y_train,
            X_val,
            y_val,
            num_class,
            metrics,
            metrics_proba,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    else:
        raise ValueError(
            "kind must be in the following : ['regression', 'binary', 'multiclass']"
        )
    return result_df


def benchmark_cv(
    kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_class: int,
    cv: int = 5,
    metrics: List[Any] = None,
    metrics_proba: List[Any] = None,
    metrics_kwargs: dict = None,
    sort_by: str = None,
    is_smaller_better: bool = True,
    plot: bool = True,
    folder_path: str = None,
):
    if kind in ["reg", "regression"]:
        result_df = regression_cv(
            X_train,
            y_train,
            cv,
            metrics,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    elif kind in ["bin", "binary"]:
        result_df = binary_cv(
            X_train,
            y_train,
            cv,
            metrics,
            metrics_proba,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    elif kind in ["multi", "multiclass"]:
        result_df = multiclass_cv(
            X_train,
            y_train,
            num_class,
            cv,
            metrics,
            metrics_proba,
            metrics_kwargs,
            sort_by,
            is_smaller_better,
            plot,
            folder_path,
        )
    return result_df
