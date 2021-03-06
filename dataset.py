import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pandas import DataFrame
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from typing import Callable, List, Any


def parallel_map_df(func: Callable[[DataFrame, Any], DataFrame], num_cores=cpu_count()):
    """
    Python decorator that must be wrapped to a function that receive as first input a pandas Dataframe.
    Apply a trasformation on all the dataframe in parallel using the multiprocessing module

    Example:
        @parallel_map_df
        def sum_of_cols(df):
            df['c'] = df['a'] + df['b']
            return df


        @parallel_map_df(num_cores=16)
        def sum_of_cols(df):
            df['c'] = df['a'] + df['b']
            return df
    """

    def wrapper(df: DataFrame, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        partitions = np.linspace(0, len(df) - 1, num_cores + 1).round().astype('int')
        df_split = [df.iloc[partitions[i]:partitions[i + 1]] for i in range(len(partitions) - 1)]
        pool = Pool(num_cores)
        df = pd.concat(pool.map(partial_func, df_split))
        pool.close()
        pool.join()
        return df

    return wrapper


def _cramerv(col1: np.array, col2: np.array):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p_value = chi2_contingency(confusion_matrix)[0:2]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if not phi2corr == 0 else 0, p_value


def cramerv(dataset: DataFrame, cols: List[str]) -> np.matrix:
    """
    Apply the cramerv test on each pair of columns. The cramerv test is used to
    measure in [0,1] the correlation between categorical variables
    :param dataset: Dataframe where the correlation is measured
    :param cols: the columns of dataset where measure the correlation
    :return: the correlation matrix of cramerv on the specified columns of dataset
    """
    matr = np.matrix([[0] * len(cols)] * len(cols), dtype='float')
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if i == j:
                matr[i, j] = 1
            else:
                corr, pvalue = _cramerv(dataset[cols[i]], dataset[cols[j]])
                matr[i, j] = corr
                matr[j, i] = corr
    return matr


def _fill_na(df, column, model):
    na_rows = df[column].isna()
    train, test = df[~na_rows], df[na_rows].drop(columns=column)
    try:
        model.fit(train.drop(columns=column).values, train[column].values)
    except ValueError as e:
        if str(e) == "Unknown label type: 'continuous'":
            model.fit(train.drop(columns=column).values, train[column].values.astype('int'))
    del train
    new_values = model.predict(test.values)
    del model
    test[column] = new_values
    df.loc[test.index, column] = test[column]
    del test
    return df


def _cat_to_int(df, targetcol):
    strcols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for strcol in strcols:
        df[strcol] = pd.to_numeric(df[strcol].astype('category').cat.codes, downcast='integer')
    if targetcol in strcols:
        df[targetcol] = df[targetcol].replace(-1, np.NaN)
    return df


def _fill_quantitative_with_mean(df, quant_cols):
    df[quant_cols] = df[quant_cols].fillna(df[quant_cols].mean())
    return df


def fill_na_as_prediction(df: pd.DataFrame, cols_na: List[str], model=None):
    """
    Fill na of column as prediction task.
    df category columns must be of type
    string, bool or category, while numeric of
    any other type (int*, float*, uint*).
    When predict the other na columns
    are filled temporally with mode/mean
    :param df: a pandas dataframe
    :param cols_na: The columns to fill
    :param model: By default it is a sklearn RandomForestClassifier
    :return: a new dataframe with na filled
    """
    quantitative_na_cols = df.loc[:, df.isna().any()] \
        .select_dtypes(exclude=['category', 'object', 'bool']) \
        .columns.tolist()
    if model is None:
        model = RandomForestClassifier(n_jobs=-1)
    for col_na in cols_na:
        if col_na in quantitative_na_cols:
            quantitative_na_cols.remove(col_na)
        tmp = df.copy()
        tmp = _fill_quantitative_with_mean(tmp, quantitative_na_cols)
        tmp = _cat_to_int(tmp, col_na)
        tmp = _fill_na(tmp, col_na, model)
        if col_na in df.select_dtypes(include=['category', 'object']).columns:
            df[col_na] = tmp[col_na].astype('category')
        else:
            df[col_na] = tmp[col_na]
        del tmp
    return df


__all__ = ['fill_na_as_prediction', 'cramerv', 'parallel_map_df']
