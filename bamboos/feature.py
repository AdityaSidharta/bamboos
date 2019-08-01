from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from bamboos.utils.dataframe import insert_df


def fit_stats(
    input_df: pd.DataFrame,
    stat_cols: List[str],
    target_cols: List[str],
    metric_cols: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    for col_name in ["stat_cols", "target_cols", "metric_cols", "default"]:
        assert (
            col_name not in stat_cols
        ), "Please don't use {} as a Stats column name".format(
            col_name
        )
        assert (
            col_name not in target_cols
        ), "Please don't use {} as a Target column name".format(
            col_name
        )
    df = input_df.copy()

    df = df[stat_cols + target_cols]
    if not metric_cols:
        metric_cols = ["mean", "median", "std", "min", "max"]
    if isinstance(metric_cols, dict):
        assert set(stat_cols) == set(metric_cols)

    stats_encoder = dict()  # type: dict
    for stat_col in stat_cols:
        stats_encoder[stat_col] = dict()
        if isinstance(metric_cols, dict):
            stat_metric_cols = metric_cols[stat_col]
        else:
            stat_metric_cols = metric_cols
        for target_col in target_cols:
            agg_df = (
                df.groupby(stat_col)[target_col].agg(stat_metric_cols).reset_index()
            )

            default_df = pd.DataFrame(
                [df[target_col].agg(stat_metric_cols)]
            ).reset_index(drop=True)
            default_df[stat_col] = "default"
            default_df = default_df[[stat_col] + stat_metric_cols]

            full_agg_df = agg_df.append(default_df)

            stat_colname = [
                "{}_{}_{}".format(stat_col, target_col, metrics_col)
                for metrics_col in stat_metric_cols
            ]
            full_agg_df.columns = [stat_col] + stat_colname
            stats_encoder[stat_col][target_col] = full_agg_df

    stats_encoder["stat_cols"] = stat_cols
    stats_encoder["target_cols"] = target_cols
    stats_encoder["metric_cols"] = metric_cols

    result_df = transform_stats(input_df, stats_encoder)
    return result_df, stats_encoder


def transform_stats(
    input_df: pd.DataFrame, stats_encoder_dict: Dict[str, Any]
) -> pd.DataFrame:
    stat_cols = stats_encoder_dict["stat_cols"]
    target_cols = stats_encoder_dict["target_cols"]
    metric_cols = stats_encoder_dict["metric_cols"]
    result_df = input_df.copy()

    # pylint: disable=cell-var-from-loop
    for stat_col in stat_cols:
        if isinstance(metric_cols, dict):
            stat_metric_cols = metric_cols[stat_col]
        else:
            stat_metric_cols = metric_cols
        for target_col in target_cols[::-1]:
            stat_col_idx = result_df.columns.get_loc(stat_col) + 1
            stat_colname = [
                "{}_{}_{}".format(stat_col, target_col, metrics_col)
                for metrics_col in stat_metric_cols
            ]

            small_df = result_df[[stat_col]].copy()
            small_df[stat_col] = small_df[stat_col].apply(
                lambda x: "default"
                if x not in stats_encoder_dict[stat_col][target_col][stat_col].tolist()
                else x
            )
            agg_df = small_df.merge(
                stats_encoder_dict[stat_col][target_col],
                how="left",
                on=stat_col,
                validate="m:1",
            )[stat_colname]
            result_df = insert_df(result_df, agg_df, stat_col_idx)
    return result_df


def fit_rolling(input_df, grouper, colname, n, fn, fn_name=None, datetime_colname=None):
    df = input_df.copy()
    if datetime_colname is not None:
        train_df = input_df.copy().sort_values([grouper, datetime_colname])
        test_df = train_df.copy()
    else:
        train_df = input_df.copy().sort_values([grouper])
        test_df = train_df.copy()

    if fn_name is not None:
        rolling_colname = "{}_{}_rolling{}_{}".format(colname, grouper, fn_name, n)
    else:
        rolling_colname = "{}_{}_rolling{}_{}".format(colname, grouper, fn, n)

    train_df[rolling_colname] = train_df.groupby(grouper)[colname].apply(
        lambda x: x.shift(1).rolling(n).agg(fn)
    )
    test_df[rolling_colname] = test_df.groupby(grouper)[colname].apply(
        lambda x: x.rolling(n).agg(fn)
    )

    test_df = test_df.reset_index(drop=True)
    test_df = test_df.drop_duplicates([grouper], keep="last")[
        [grouper, rolling_colname]
    ]

    train_df = train_df.sort_index()
    df[rolling_colname] = train_df[rolling_colname]
    rolling_encoder = {
        "test_df": test_df,
        "grouper": grouper,
        "colname": colname,
        "columns": df.columns,
    }

    return df, rolling_encoder


def transform_rolling(input_df, encoder):
    """
    Performing Rolling Transform function on test_data. Look at the explanation above for more details.
    """
    df = input_df.copy()
    grouper = encoder["grouper"]
    test_df = encoder["test_df"]
    columns = encoder["columns"]
    df = df.merge(test_df, on=grouper, how="left", validate="m:1")

    assert np.all(df.columns == columns)
    return df


def fit_sampler(
    input_df,
    grouper,
    colname,
    fn,
    datetime_colname,
    resampler="D",
    fn_name=None,
    min_obs=None,
):
    """
    Performing Sampling function.
    Sampling function is creating by aggregating all the observations in `colname` column from the previous day.
    The aggregation uses `fn` as the aggregation function, and this aggregation is grouped using the `grouper`

    For example,
    if grouper = `driver_id`, colname=`is_booking_completed`, datetime_colname='event_timestamp`,
    fn = `sum`

    The value `is_booking_completed_driver_id_sum` for a specific driver_id on 2015-05-16 will be the sum of the
    number of booking completed by that driver on the previous day (2015-05-15)
    """
    df = input_df.copy()
    grouper_df = input_df.copy()
    if isinstance(min_obs, int):
        counts = grouper_df[grouper].value_counts()
        grouper_df = grouper_df[
            ~grouper_df[grouper].isin(counts[counts < min_obs].index)
        ]

    df["tmp_" + datetime_colname] = pd.to_datetime(df[datetime_colname])
    grouper_df[datetime_colname] = pd.to_datetime(grouper_df[datetime_colname])

    grouper_df = grouper_df.set_index(datetime_colname)
    result = grouper_df.groupby(grouper)[colname].resample(resampler).agg(fn)

    train_result = result.groupby(grouper).shift(1).reset_index()
    test_result = result.groupby(grouper).last().reset_index()

    if fn_name is not None:
        train_result.columns = [
            grouper,
            "tmp_" + datetime_colname,
            "{}_{}_{}".format(colname, grouper, fn_name),
        ]
        test_result.columns = [grouper, "{}_{}_{}".format(colname, grouper, fn_name)]
    else:
        train_result.columns = [
            grouper,
            "tmp_" + datetime_colname,
            "{}_{}_{}".format(colname, grouper, fn),
        ]
        test_result.columns = [grouper, "{}_{}_{}".format(colname, grouper, fn)]

    df = df.merge(
        train_result,
        on=[grouper, "tmp_" + datetime_colname],
        how="left",
        validate="m:1",
    )
    df = df.drop(columns=["tmp_" + datetime_colname])

    return (
        df,
        {
            "grouper": grouper,
            "colname": colname,
            "datetime_colname": datetime_colname,
            "fn": fn,
            "fn_name": fn_name,
            "test_result": test_result,
            "columns": df.columns,
        },
    )


def transform_sampler(input_df, encoder):
    """
    Performing Transform on Sampler Function for test_data
    """
    df = input_df.copy()
    test_result = encoder["test_result"]
    grouper = encoder["grouper"]
    columns = encoder["columns"]
    df = df.merge(test_result, on=grouper, how="left", validate="m:1")
    assert np.all(df.columns == columns)
    return df
