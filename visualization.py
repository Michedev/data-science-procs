import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def pca_plot_train_test(train: pd.DataFrame, test: pd.DataFrame, **lmplot_args):
    """
    Make a 2-PCA plot with different colors for train and test.
    Useful to understand the train and test distribution.
    NOTE: train and test must have the same number of columns
    :param train: train dataframe
    :param test: test dataframe
    :param lmplot_args: args for seaborn lmplot
    :return: the 2-PCA df
    """
    assert len(train.columns) == len(test.columns)
    train_test = pd.concat([train, test], copy=False)
    train_test.reset_index(inplace=True, drop=True)
    train_test['_is_train_'] = train_test.index < len(train)
    dfpca = pca_plot(df=train_test, target='_is_train_', **lmplot_args)
    del train_test
    return dfpca


def pca_plot(df, target, **lmplot_args):
    """
    Do 2-PCA to df and plot a scatter plot with hue set to target
    :param df: dataframe
    :param target: name of the column where hue is applied
    :param lmplot_args: args for seaborn lmplot
    :return: the 2-PCA df
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop(columns=target).values)
    df_pca = pd.DataFrame(X_pca, columns=['a', 'b'])
    df_pca[target] = df[target]
    sns.lmplot(x='a', y='b', data=df_pca, hue=target, **lmplot_args)
    plt.show()
    return df_pca


__all__ = ['pca_plot', 'pca_plot_train_test']
