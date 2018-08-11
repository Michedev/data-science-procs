from cmath import log

import pandas as pd
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from typing import Iterable, Union, Tuple
from utils import parse_X_y_input


def validate_param(param_name: str, param_values: Iterable, train: Union[DataFrame, np.ndarray],
                   target: Union[str, int, Series, np.array], model=None,
                   scoring='accuracy', cv=None):
    X, y = parse_X_y_input(train, target)
    assert hasattr(scoring, '__name__') and not isinstance(scoring, str),\
        'Scoring object must have __name__ attribute if not string'
    if model is None:
        model = RandomForestClassifier()
    scorename = scoring if isinstance(scoring, str) else scoring.__name__
    train_scores, valid_scores = validation_curve(model, X, y, param_name, param_values, scoring=scoring, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with " + model.__class__.__name__)
    plt.xlabel(param_name)
    plt.ylabel(scorename)
    plt.xticks(param_values)
    plt.semilogx(param_values, train_scores_mean, label="Training score", color='#78A8F3')
    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="#78A8F3")
    plt.semilogx(param_values, valid_scores_mean, label="Cross-validation score",
                 color="#F37878")
    plt.fill_between(param_values, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.2,
                     color="#F37878")
    plt.legend(loc="best")
    plt.show()
    return train_scores, valid_scores


def _take_first_k(seq: np.array, k, keyf=lambda x: x) -> np.array:
    n = len(seq)
    if k > log(n):
        return sorted(seq, key=keyf)[:k]
    first_k = []
    for _ in range(k):
        max_el = max(seq, key=keyf)
        first_k.append(max_el)
        seq[max_el[0]][1] = -1  # to avoid remotion
    return first_k


def adversial_validation(train: DataFrame, test: DataFrame, perc_val=0.3, model=None) -> Tuple[DataFrame, DataFrame]:
    """
    Do an adversial validation on the input datasets. Basically the adversial validation consists on finding the most similar
    validation set to the test test. You can find more info here: http://manishbarnwal.com/blog/2017/02/15/introduction_to_adversarial_validation/
    Attention: the columns on train and test must be in a numeric form (No category, string, boolean)
    Note: train and test must have the same columns
    :return a tuple (train, validation)
    """
    if model is None:
        model = RandomForestClassifier()
    train_orig = train
    assert train.columns.isin(test.columns).all() and train.shape[1] == test.shape[1]
    train['__target'] = 0
    test['__target'] = 1
    dataset = pd.concat([train, test])
    X, y = dataset.drop('__target', axis=1), dataset['__target']
    model.fit(X, y)
    del train['__target']
    del test['__target']
    p = list(enumerate(map(itemgetter(1), model.predict_proba(train))))
    k = int(round(perc_val * len(train)))
    higher_proba = _take_first_k(p, k, itemgetter(1))
    newtest = train_orig.iloc[list(map(itemgetter(0), higher_proba))]
    newtrain = train_orig.drop(newtest.index)
    return newtrain, newtest


__all__ = ['validate_param', 'adversial_validation']
