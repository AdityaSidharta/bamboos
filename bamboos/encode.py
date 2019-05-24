import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder, OneHotEncoder, BinaryEncoder


def fit_label(input_df, cols, na_value=None):
    df = input_df.copy()

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    encoder = OrdinalEncoder(cols)
    encoder = encoder.fit(df)
    for idx in range(len(encoder.mapping)):
        encoder.mapping[idx]["mapping"].loc[np.nan] = -2

    result_df = encoder.transform(df)

    for col in cols:
        result_df[col] = result_df[col].replace({-1: 0, -2: 0})

    model = {"encoder": encoder, "cols": cols, "na_value": na_value}
    return result_df, model


def transform_label(input_df, model):
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
    return result_df


def fit_onehot(input_df, cols, na_value=None):
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

    model = {"encoder": encoder, "cols": cols, "na_value": na_value, "drop_cols": drop_cols}
    return result_df, model


def transform_onehot(input_df, model):
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


def fit_binary(input_df, cols, na_value=None):
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


def transform_binary(input_df, model):
    df = input_df.copy()

    encoder = model["encoder"]
    cols = model["cols"]
    na_value = model["na_value"]

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    result_df = encoder.transform(df)

    return result_df


def fit_categorical(input_df, cols, na_value=None, max_onehot=10, max_binary=1000):
    df = input_df.copy()

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    onehot_cols = []
    label_cols = []
    binary_cols = []

    for col in cols:
        col_values = df[col].values
        cardinality = len(np.unique(col_values[~pd.isnull(col_values)]))
        if cardinality < max_onehot:
            onehot_cols.append(col)
        elif cardinality < max_binary:
            label_cols.append(col)
        else:
            binary_cols.append(col)

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


def transform_categorical(input_df, model):
    df = input_df.copy()

    onehot_model = model["onehot_model"]
    label_model = model["label_model"]
    binary_model = model["binary_model"]
    cols = model["cols"]
    na_value = model["na_value"]

    if na_value is not None:
        for col in cols:
            df[col] = df[col].replace({na_value: np.nan})

    df = transform_onehot(df, onehot_model)
    df = transform_label(df, label_model)
    result_df = transform_binary(df, binary_model)

    return result_df