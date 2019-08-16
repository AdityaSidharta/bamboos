from typing import Any, List

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, OneHotEncoder, BinaryEncoder


def fit_label(input_df: pd.DataFrame, cols: List[str], na_value: Any = None):
    """
    Creates the label encoder by fitting it through the given DataFrame
    NaN values and Special value specified under `na_value` in the DataFrame will be encoded as unseen value.
    Args:
        input_df: DataFrame used to fit the encoder
        cols: List of categorical columns to be encoded
        na_value: Default null value for DataFrame

    Returns:
        result_df: encoded input_df DataFrame
        model : encoder model to be passed to `transform_label` method
    """
    df = input_df.copy()

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    encoder = OrdinalEncoder(cols=cols)
    encoder = encoder.fit(df)
    for idx in range(len(encoder.mapping)):
        encoder.mapping[idx]["mapping"].loc[np.nan] = -2

    result_df = encoder.transform(df)

    for col in cols:
        result_df[col] = result_df[col].replace({-1: 0, -2: 0})
        result_df[col] = result_df[col].astype(int)

    model = {"encoder": encoder, "cols": cols, "na_value": na_value}
    return result_df, model


def transform_label(input_df: pd.DataFrame, model: Any):
    """
    Perform Label encoding to the given DataFrame using previously fitted encoder
    Previously unseen value, NaN values, and Special value specified under `na_value` in the DataFrame
    will be encoded as unseen value.
    Args:
        input_df: DataFrame to be encoded
        model: Fitted label Encoder Class
    Returns:
        result_df: encoded input_df DataFrame
    """
    df = input_df.copy()

    encoder = model["encoder"]
    cols = model["cols"]
    na_value = model["na_value"]

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    result_df = encoder.transform(df)

    for col in cols:
        result_df[col] = result_df[col].replace({-1: 0, -2: 0})
        result_df[col] = result_df[col].astype(int)

    return result_df


def fit_onehot(input_df: pd.DataFrame, cols: List[str], na_value: Any = None):
    """
    Creates the One-hot encoder by fitting it through the given DataFrame
    NaN values and Special value specified under `na_value` in the DataFrame will be encoded as unseen value.
    Args:
        input_df: DataFrame used to fit the encoder
        cols: List of categorical columns to be encoded
        na_value: Default null value for DataFrame

    Returns:
        result_df: encoded input_df DataFrame
        model : encoder model to be passed to `transform_onehot` method
    """
    df = input_df.copy()

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    drop_cols = ["{}_nan".format(col) for col in cols]

    encoder = OneHotEncoder(cols=cols, use_cat_names=True)
    encoder = encoder.fit(df)

    result_df = encoder.transform(df)

    for drop_col in drop_cols:
        if drop_col in result_df.columns:
            result_df = result_df.drop(columns=[drop_col])

    model = {
        "encoder": encoder,
        "cols": cols,
        "na_value": na_value,
        "drop_cols": drop_cols,
    }
    return result_df, model


def transform_onehot(input_df: pd.DataFrame, model: Any):
    """
    Perform One-hot encoding to the given DataFrame using previously fitted encoder
    Previously unseen value, NaN values, and Special value specified under `na_value` in the DataFrame
    will be encoded as unseen value.
    Args:
        input_df: DataFrame to be encoded
        model: Fitted One-hot Encoder Class
    Returns:
        result_df: encoded input_df DataFrame
    """
    df = input_df.copy()

    encoder = model["encoder"]
    cols = model["cols"]
    na_value = model["na_value"]
    drop_cols = model["drop_cols"]

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    result_df = encoder.transform(df)

    for drop_col in drop_cols:
        if drop_col in result_df.columns:
            result_df = result_df.drop(columns=[drop_col])

    return result_df


def fit_binary(input_df: pd.DataFrame, cols: List[str], na_value: Any = None):
    """
    Creates the binary encoder by fitting it through the given DataFrame.
    NaN values and Special value specified under `na_value` in the DataFrame will be encoded as unseen value.
    Args:
        input_df: DataFrame used to fit the encoder
        cols: List of categorical columns to be encoded
        na_value: Default null value for DataFrame

    Returns:
        result_df: encoded input_df DataFrame
        model : encoder model to be passed to `transform_binary` method
    """
    df = input_df.copy()

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    encoder = BinaryEncoder(cols=cols, drop_invariant=True)
    encoder = encoder.fit(df)
    for idx in range(len(encoder.base_n_encoder.ordinal_encoder.mapping)):
        encoder.base_n_encoder.ordinal_encoder.mapping[idx]["mapping"].loc[np.nan] = -2

    result_df = encoder.transform(df)

    model = {"encoder": encoder, "cols": cols, "na_value": na_value}
    return result_df, model


def transform_binary(input_df: pd.DataFrame, model: Any):
    """
    Perform Binary encoding to the given DataFrame using previously fitted encoder.
    Previously unseen value, NaN values, and Special value specified under `na_value` in the DataFrame
    will be encoded as unseen value.
    Args:
        input_df: DataFrame to be encoded
        model: Fitted Binary Encoder Class
    Returns:
        result_df: encoded input_df DataFrame
    """
    df = input_df.copy()

    encoder = model["encoder"]
    cols = model["cols"]
    na_value = model["na_value"]

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    result_df = encoder.transform(df)

    return result_df


def fit_categorical(
    input_df: pd.DataFrame,
    cols: List[str],
    na_value: Any = None,
    max_onehot: int = 10,
    max_binary: int = 1000,
):
    """
    Perform Automated Encoding for all categorical columns.
    All columns with low cardinality ( less than binary_limit unique values ) will be encoded using Label encoder
    All columns with high cardinality ( More than binary_limit unique values ) will be encoded using
    Categorical encoder
    Args:
        input_df: DataFrame to be encoded
        cols: List of categorical columns to be encoded
        na_value: Default na value for DataFrame
        max_binary: number of cardinality. If the column has smaller cardinality than max_binary and bigger t
        han max_onehot, will be encoded using LabelEncoder. Else, will be encoded using BinaryEncoder.
        max_onehot: number of cardinality. This value must be smaller than min_binary.
        If the column has smaller cardinality to min_onehot, the column will be
         encoded using OneHotEncoder.
    Returns:
        result_df: encoded input_df DataFrame
        model : encoder model to be passed to `transform_categorical` method
    """
    df = input_df.copy()

    onehot_cols = []
    label_cols = []
    binary_cols = []

    for col in cols:
        col_values = df[col].values
        cardinality = len(np.unique(col_values[~pd.isnull(col_values)]))
        if cardinality < max_onehot:
            onehot_cols.append(col)
        elif cardinality < max_binary:
            binary_cols.append(col)
        else:
            label_cols.append(col)

    df, onehot_model = fit_onehot(df, onehot_cols, na_value)
    df, label_model = fit_label(df, label_cols, na_value)
    result_df, binary_model = fit_binary(df, binary_cols, na_value)

    model = {
        "onehot_model": onehot_model,
        "label_model": label_model,
        "binary_model": binary_model,
        "onehot_cols": onehot_cols,
        "label_cols": label_cols,
        "binary_cols": binary_cols,
        "cols": cols,
        "na_value": na_value,
        "max_onehot": max_onehot,
        "max_binary": max_binary,
    }
    return result_df, model


def transform_categorical(input_df: pd.DataFrame, model: Any):
    """
    Perform Categorical encoding to the given DataFrame using previously fitted encoder
    All columns with low cardinality ( less than binary_limit unique values ) will be encoded using Label encoder
    All columns with high cardinality ( More than binary_limit unique values ) will be encoded using
    Categorical encoder
    Args:
        input_df: DataFrame to be encoded
        model: Fitted Categorical Encoder Class
    Returns:
        result_df: encoded input_df DataFrame
    """
    df = input_df.copy()

    onehot_model = model["onehot_model"]
    label_model = model["label_model"]
    binary_model = model["binary_model"]

    df = transform_onehot(df, onehot_model)
    df = transform_label(df, label_model)
    result_df = transform_binary(df, binary_model)

    return result_df
