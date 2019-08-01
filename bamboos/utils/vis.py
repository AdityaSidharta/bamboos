import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(df: pd.DataFrame, x: str, y: str, filename: str = None):
    """
    Perform Box Plot on a given dataframe, with options to save the plot in a given path

    Args:
        df (pd.DataFrame): DataFrame to be plotted
        x (str): Column name within the dataframe, to be used as x-axis
        y (str): Column name within the dataframe, to be used as y-axis
        filename (str): Path where the file should be saved, None if we only want to show the plot

    Returns:
        (None): PNG file if filename contains the path, Plotted figure if filename is None
    """
    fig, ax = plt.subplots(figsize=(25, 12))
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
    else:
        fig.show()


def barplot(df: pd.DataFrame, x: str, y: str, hue: str = None, filename: str = None):
    """
    Perform Bar plot on a given dataframe, with options to save the plot in a given path

    Args:
        df (pd.DataFrame): DataFrame to be plotted
        x (str): Column name within the dataframe, to be used as x-axis
        y (str): Column name within the dataframe, to be used as y-axis
        hue (str): Column name within the dataframe, to be used as hue
        filename (path): Path where the file should be saved, None if we only want to show the plot

    Returns:
        (None): PNG file if filename contains the path, Plotted figure if filename is None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax)
    if filename:
        fig.savefig(filename)
    else:
        fig.show()
