import pandas as pd
import numpy as np


def parse_X_y_input(train, y):
    if isinstance(train, pd.DataFrame) and isinstance(y, str):
        return train.drop(columns=y), train[y]
    elif isinstance(train, pd.DataFrame) and isinstance(y, int):
        return train.loc[:, np.arange(train.shape[1]) != y], train.iloc[:, y]
    elif isinstance(train, np.ndarray) and isinstance(y, int):
        return np.delete(train, y, axis=1), train[:, y]
    elif isinstance(train, (pd.DataFrame, np.ndarray)) and isinstance(y, (pd.Series, np.array)):
        return train, y
    raise TypeError(f"Combination of types to parse not recognized: {type(train)}, {type(y)}")