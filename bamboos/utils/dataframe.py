import pandas as pd


def locate_col(df: pd.DataFrame, col_name: str):
    """
    Return index after col_name within the dataframe df
    Args:
        df: DataFrame of interest
        col_name: Name of columns within the DataFrame

    Returns:
        (int): index of the column within the dataframe
    """
    idx = df.columns.get_loc(col_name) + 1
    return idx


def insert_df(outer_df: pd.DataFrame, inner_df: pd.DataFrame, loc: int):
    """
    Insert `inner_df` into `outer_df` at specified index, `loc`.

    Args:
        outer_df (pd.DataFrame): DataFrame which will be inserted by another DataFrame
        inner_df (pd.DataFrame): DataFrame to be inserted
        loc (int): location index of insertion

    Returns:
        (pd.DataFrame), `outer_df` with `inner_df` inserted in between according to the specified index
    """
    return pd.concat([outer_df.iloc[:, :loc], inner_df, outer_df.iloc[:, loc:]], axis=1)
