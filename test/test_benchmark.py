import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_boston, load_breast_cancer, load_digits
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
from sklearn.model_selection import train_test_split

from bamboos.utils.metrics.sklearn import pr_auc_score
from bamboos.benchmark import (
    get_default_metric,
    bm,
    bm_cv,
    aggregate,
    sort,
    plot_save,
    plot_save_cv,
    regression,
    binary,
    multiclass,
    regression_cv,
    binary_cv,
    multiclass_cv,
    benchmark,
    benchmark_cv,
)
from bamboos.utils.model.model_zoo import (
    binary_model_dict,
    multiclass_model_dict,
    regression_model_dict,
)


@pytest.fixture()
def regression_var():
    features, target = load_boston(return_X_y=True)
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    regression_var = {
        "X_train": X_train[:130],
        "X_val": X_test[:130],
        "y_train": y_train[:130],
        "y_val": y_test[:130],
    }
    return regression_var


@pytest.fixture()
def binary_var():
    features, target = load_breast_cancer(return_X_y=True)
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    binary_var = {
        "X_train": X_train[:130],
        "X_val": X_test[:130],
        "y_train": y_train[:130],
        "y_val": y_test[:130],
    }
    return binary_var


@pytest.fixture()
def multiclass_var():
    features, target = load_digits(n_class=10, return_X_y=True)
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    multiclass_var = {
        "X_train": X_train[:130],
        "X_val": X_test[:130],
        "y_train": y_train[:130],
        "y_val": y_test[:130],
    }
    return multiclass_var


def test_get_default_metric_regression():
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "regression"
    )

    assert metrics == [
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        explained_variance_score,
    ]
    assert metrics_proba == []
    assert metrics_kwargs == {}
    assert sort_by == mean_absolute_error.__name__
    assert ascending


def test_get_default_metric_binary():
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "binary"
    )

    assert metrics == [accuracy_score, recall_score, precision_score, f1_score]
    assert metrics_proba == [roc_auc_score, pr_auc_score]
    assert metrics_kwargs == {}
    assert sort_by == f1_score.__name__
    assert not ascending


def test_get_default_metric_multiclass():
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "multiclass"
    )

    assert metrics == [accuracy_score, recall_score, precision_score, f1_score]
    assert metrics_proba == [roc_auc_score]
    assert metrics_kwargs == {
        "recall_score": {"average": "macro"},
        "precision_score": {"average": "macro"},
        "f1_score": {"average": "macro"},
        "roc_auc_score": {"average": "macro"},
    }
    assert sort_by == f1_score.__name__
    assert not ascending


def test_get_default_metric_error():
    with pytest.raises(ValueError):
        get_default_metric("wrongtype")


def test_benchmark_regression(regression_var):
    X_train, X_val = regression_var["X_train"], regression_var["X_val"]
    y_train, y_val = regression_var["y_train"], regression_var["y_val"]
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "regression"
    )
    result = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        metrics_proba,
        metrics_kwargs,
        regression_model_dict(),
    )

    assert list(result.columns) == [
        "model_name",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
        "explained_variance_score",
    ]
    assert result.shape == (13, 5)
    assert list(result["model_name"]) == [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNeighborsRegressor",
        "DecisionTreeRegressor",
        "AdaBoostRegressor",
        "BaggingRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "RandomForestRegressor",
        "XGBoost",
        "LightGBM",
    ]
    assert result["mean_absolute_error"].mean() < 8
    assert result["mean_squared_error"].mean() < 100
    assert result["r2_score"].mean() > 0.5
    assert result["explained_variance_score"].mean() > 0.5


def test_benchmark_binary(binary_var):
    X_train, X_val = binary_var["X_train"], binary_var["X_val"]
    y_train, y_val = binary_var["y_train"], binary_var["y_val"]
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "binary"
    )
    result = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        metrics_proba,
        metrics_kwargs,
        binary_model_dict(),
    )

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
        "pr_auc_score",
    ]
    assert result.shape == (11, 7)
    assert list(result["model_name"]) == [
        "LogisticRegression",
        "GaussianNB",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "AdaBoostClassifier",
        "BaggingClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "RandomForestClassifier",
        "XGBoost",
        "LightGBM",
    ]
    assert result["accuracy_score"].mean() > 0.85
    assert result["recall_score"].mean() > 0.85
    assert result["precision_score"].mean() > 0.85
    assert result["f1_score"].mean() > 0.85
    assert result["roc_auc_score"].mean() > 0.85
    assert result["pr_auc_score"].mean() > 0.85


def test_benchmark_multiclass(multiclass_var):
    X_train, X_val = multiclass_var["X_train"], multiclass_var["X_val"]
    y_train, y_val = multiclass_var["y_train"], multiclass_var["y_val"]
    num_class = np.unique(y_train).size
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "multiclass"
    )
    result = bm(
        X_train,
        y_train,
        X_val,
        y_val,
        metrics,
        metrics_proba,
        metrics_kwargs,
        multiclass_model_dict(num_class=num_class),
    )

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
    ]
    assert result.shape == (8, 6)
    assert list(result["model_name"]) == [
        "LogisticRegression",
        "GaussianNB",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "RandomForestClassifier",
        "XGBoost",
        "LightGBM",
    ]
    assert result["accuracy_score"].mean() > 0.75
    assert result["recall_score"].mean() > 0.75
    assert result["precision_score"].mean() > 0.75
    assert result["f1_score"].mean() > 0.75
    assert result["roc_auc_score"].mean() > 0.75


def test_benchmarkcv_regression(regression_var):
    X_train, y_train = regression_var["X_train"], regression_var["y_train"]
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "regression"
    )
    cv = 3

    result = bm_cv(
        X_train,
        y_train,
        cv,
        metrics,
        metrics_proba,
        metrics_kwargs,
        regression_model_dict(),
    )

    assert list(result.groupby("cv_idx").size().values) == [13] * cv
    assert list(result.groupby("model_name").size().values) == [3] * len(
        regression_model_dict()
    )
    assert result["mean_absolute_error"].mean() < 8
    assert result["mean_squared_error"].mean() < 100
    assert result["r2_score"].mean() > 0.5
    assert result["explained_variance_score"].mean() > 0.5


def test_benchmarkcv_binary(binary_var):
    X_train, y_train = binary_var["X_train"], binary_var["y_train"]
    metrics, metrics_proba, metrics_kwargs, sort_by, ascending = get_default_metric(
        "binary"
    )
    cv = 3

    result = bm_cv(
        X_train,
        y_train,
        cv,
        metrics,
        metrics_proba,
        metrics_kwargs,
        binary_model_dict(),
    )

    assert list(result.groupby("cv_idx").size().values) == [11] * cv
    assert list(result.groupby("model_name").size().values) == [3] * len(
        binary_model_dict()
    )
    assert result["accuracy_score"].mean() > 0.85
    assert result["recall_score"].mean() > 0.85
    assert result["precision_score"].mean() > 0.85
    assert result["f1_score"].mean() > 0.85
    assert result["roc_auc_score"].mean() > 0.85
    assert result["pr_auc_score"].mean() > 0.85


def test_regression(regression_var):
    X_train, X_val = regression_var["X_train"], regression_var["X_val"]
    y_train, y_val = regression_var["y_train"], regression_var["y_val"]
    result = regression(X_train, y_train, X_val, y_val)

    assert list(result.columns) == [
        "model_name",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
        "explained_variance_score",
    ]
    assert result.shape == (13, 5)
    assert set(result["model_name"]) == {
        "AdaBoostRegressor",
        "BaggingRegressor",
        "DecisionTreeRegressor",
        "ElasticNet",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "KNeighborsRegressor",
        "Lasso",
        "LightGBM",
        "LinearRegression",
        "RandomForestRegressor",
        "Ridge",
        "XGBoost",
    }
    assert result["mean_absolute_error"].mean() < 7
    assert result["mean_squared_error"].mean() < 100


def test_binary(binary_var):
    X_train, X_val = binary_var["X_train"], binary_var["X_val"]
    y_train, y_val = binary_var["y_train"], binary_var["y_val"]
    result = binary(X_train, y_train, X_val, y_val)

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
        "pr_auc_score",
    ]
    assert result.shape == (11, 7)
    assert set(result["model_name"]) == {
        "AdaBoostClassifier",
        "BaggingClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "LightGBM",
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost",
    }
    assert result["accuracy_score"].mean() > 0.8
    assert result["recall_score"].mean() > 0.8
    assert result["precision_score"].mean() > 0.8
    assert result["f1_score"].mean() > 0.8
    assert result["roc_auc_score"].mean() > 0.9
    assert result["pr_auc_score"].mean() > 0.9


def test_multiclass(multiclass_var):
    X_train, X_val = multiclass_var["X_train"], multiclass_var["X_val"]
    y_train, y_val = multiclass_var["y_train"], multiclass_var["y_val"]
    num_class = np.unique(y_train).size
    result = multiclass(X_train, y_train, X_val, y_val, num_class=num_class)

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
    ]
    assert result.shape == (8, 6)
    assert set(result["model_name"]) == {
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "KNeighborsClassifier",
        "LightGBM",
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost",
    }
    assert result["accuracy_score"].mean() > 0.7
    assert result["recall_score"].mean() > 0.7
    assert result["precision_score"].mean() > 0.7
    assert result["f1_score"].mean() > 0.7
    assert result["roc_auc_score"].mean() > 0.8


def test_regression_cv(regression_var):
    X_train, y_train = regression_var["X_train"], regression_var["y_train"]
    cv = 3
    result = regression_cv(X_train, y_train, cv)

    assert list(result.columns) == [
        "model_name",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
        "explained_variance_score",
    ]
    assert result.shape == (13, 5)
    assert set(result["model_name"]) == {
        "AdaBoostRegressor",
        "BaggingRegressor",
        "DecisionTreeRegressor",
        "ElasticNet",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "KNeighborsRegressor",
        "Lasso",
        "LightGBM",
        "LinearRegression",
        "RandomForestRegressor",
        "Ridge",
        "XGBoost",
    }
    assert result["mean_absolute_error"].mean() < 10
    assert result["mean_squared_error"].mean() < 100
    assert result["r2_score"].mean() > 0.5
    assert result["explained_variance_score"].mean() > 0.5


def test_binary_cv(binary_var):
    X_train, y_train = binary_var["X_train"], binary_var["y_train"]
    cv = 3
    result = binary_cv(X_train, y_train, cv)

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
        "pr_auc_score",
    ]
    assert result.shape == (11, 7)
    assert set(result["model_name"]) == {
        "AdaBoostClassifier",
        "BaggingClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "LightGBM",
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost",
    }

    assert result["accuracy_score"].mean() > 0.8
    assert result["recall_score"].mean() > 0.8
    assert result["precision_score"].mean() > 0.8
    assert result["f1_score"].mean() > 0.8
    assert result["roc_auc_score"].mean() > 0.8
    assert result["pr_auc_score"].mean() > 0.8


def test_multiclass_cv(multiclass_var):
    X_train, y_train = multiclass_var["X_train"], multiclass_var["y_train"]
    num_class = np.unique(y_train).size
    cv = 3
    result = multiclass_cv(X_train, y_train, cv=cv, num_class=num_class)

    assert list(result.columns) == [
        "model_name",
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
    ]
    assert result.shape == (8, 6)
    assert set(result["model_name"]) == {
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GaussianNB",
        "KNeighborsClassifier",
        "LightGBM",
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBoost",
    }
    assert result["accuracy_score"].mean() > 0.7
    assert result["recall_score"].mean() > 0.7
    assert result["precision_score"].mean() > 0.7
    assert result["f1_score"].mean() > 0.7
    assert result["roc_auc_score"].mean() > 0.7


def test_benchmark(regression_var, binary_var, multiclass_var):
    X_train, X_val = regression_var["X_train"], regression_var["X_val"]
    y_train, y_val = regression_var["y_train"], regression_var["y_val"]
    result = regression(X_train, y_train, X_val, y_val)
    alt_result = benchmark('regression', X_train, y_train, X_val, y_val)
    assert list(result.columns) == list(alt_result.columns)
    assert result.shape == alt_result.shape
    assert set(result["model_name"]) == set(alt_result["model_name"])
    assert result["mean_absolute_error"].mean() < 10
    assert result["mean_squared_error"].mean() < 100
    assert result["r2_score"].mean() > 0.5
    assert result["explained_variance_score"].mean() > 0.5
    assert alt_result["mean_absolute_error"].mean() < 10
    assert alt_result["mean_squared_error"].mean() < 100
    assert alt_result["r2_score"].mean() > 0.5
    assert alt_result["explained_variance_score"].mean() > 0.5

    X_train, X_val = binary_var["X_train"], binary_var["X_val"]
    y_train, y_val = binary_var["y_train"], binary_var["y_val"]
    result = binary(X_train, y_train, X_val, y_val)
    alt_result = benchmark('binary', X_train, y_train, X_val, y_val)
    assert list(result.columns) == list(alt_result.columns)
    assert result.shape == alt_result.shape
    assert set(result["model_name"]) == set(alt_result["model_name"])
    assert result["accuracy_score"].mean() > 0.8
    assert result["recall_score"].mean() > 0.8
    assert result["precision_score"].mean() > 0.8
    assert result["f1_score"].mean() > 0.8
    assert result["roc_auc_score"].mean() > 0.8
    assert result["pr_auc_score"].mean() > 0.8
    assert alt_result["accuracy_score"].mean() > 0.8
    assert alt_result["recall_score"].mean() > 0.8
    assert alt_result["precision_score"].mean() > 0.8
    assert alt_result["f1_score"].mean() > 0.8
    assert alt_result["roc_auc_score"].mean() > 0.8
    assert alt_result["pr_auc_score"].mean() > 0.8

    X_train, X_val = multiclass_var["X_train"], multiclass_var["X_val"]
    y_train, y_val = multiclass_var["y_train"], multiclass_var["y_val"]
    num_class = np.unique(y_train).size
    result = multiclass(X_train, y_train, X_val, y_val, num_class=num_class)
    alt_result = benchmark('multiclass', X_train, y_train, X_val, y_val, num_class=num_class)
    assert result["accuracy_score"].mean() > 0.7
    assert result["recall_score"].mean() > 0.7
    assert result["precision_score"].mean() > 0.7
    assert result["f1_score"].mean() > 0.7
    assert result["roc_auc_score"].mean() > 0.7
    assert alt_result["accuracy_score"].mean() > 0.7
    assert alt_result["recall_score"].mean() > 0.7
    assert alt_result["precision_score"].mean() > 0.7
    assert alt_result["f1_score"].mean() > 0.7
    assert alt_result["roc_auc_score"].mean() > 0.7


def test_benchmark_cv(regression_var, binary_var, multiclass_var):
    cv = 3

    X_train, y_train = regression_var["X_train"], regression_var["y_train"]
    result = regression_cv(X_train, y_train, cv)
    alt_result = benchmark_cv('regression', X_train, y_train, cv=cv)
    assert list(result.columns) == list(alt_result.columns)
    assert result.shape == alt_result.shape
    assert set(result["model_name"]) == set(alt_result["model_name"])
    assert result["mean_absolute_error"].mean() < 10
    assert result["mean_squared_error"].mean() < 100
    assert result["r2_score"].mean() > 0.5
    assert result["explained_variance_score"].mean() > 0.5
    assert alt_result["mean_absolute_error"].mean() < 10
    assert alt_result["mean_squared_error"].mean() < 100
    assert alt_result["r2_score"].mean() > 0.5
    assert alt_result["explained_variance_score"].mean() > 0.5

    X_train, y_train = binary_var["X_train"], binary_var["y_train"]
    result = binary_cv(X_train, y_train, cv)
    alt_result = benchmark_cv('binary', X_train, y_train, cv=cv)
    assert list(result.columns) == list(alt_result.columns)
    assert result.shape == alt_result.shape
    assert set(result["model_name"]) == set(alt_result["model_name"])
    assert result["accuracy_score"].mean() > 0.8
    assert result["recall_score"].mean() > 0.8
    assert result["precision_score"].mean() > 0.8
    assert result["f1_score"].mean() > 0.8
    assert result["roc_auc_score"].mean() > 0.8
    assert alt_result["accuracy_score"].mean() > 0.8
    assert alt_result["recall_score"].mean() > 0.8
    assert alt_result["precision_score"].mean() > 0.8
    assert alt_result["f1_score"].mean() > 0.8
    assert alt_result["roc_auc_score"].mean() > 0.8

    X_train, y_train = multiclass_var["X_train"], multiclass_var["y_train"]
    num_class = np.unique(y_train).size
    result = multiclass_cv(X_train, y_train,  num_class=num_class, cv=cv)
    alt_result = benchmark_cv('multiclass', X_train, y_train, num_class=num_class, cv=cv)
    assert result["accuracy_score"].mean() > 0.7
    assert result["recall_score"].mean() > 0.7
    assert result["precision_score"].mean() > 0.7
    assert result["f1_score"].mean() > 0.7
    assert result["roc_auc_score"].mean() > 0.7
    assert alt_result["accuracy_score"].mean() > 0.7
    assert alt_result["recall_score"].mean() > 0.7
    assert alt_result["precision_score"].mean() > 0.7
    assert alt_result["f1_score"].mean() > 0.7
    assert alt_result["roc_auc_score"].mean() > 0.7