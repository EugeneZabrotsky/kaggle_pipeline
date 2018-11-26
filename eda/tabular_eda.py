import seaborn as sns
from functools import wraps, partial
from scipy import stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def print_column_info(df, columns):
    '''
    df - dataframe
    columns - one column or columns
    '''
    _columns = list(columns)
    # for col in columns:
    #     print()
    #     print(f'-----{col}-----')
    #     print(df[col].describe())
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        display(df[_columns].describe())


def print_skewness_kurtosis(data):
    '''
    data - array or Series
    '''
    print(f'skewness: {data.skew()}')
    print(f'kurtosis: {data.kurt()}')


def fill_with_missing(df, columns):
    '''
    QUALITATIVE columns
    '''
    df_copy = df.copy()
    for c in columns:
        df_copy[c] = df_copy[c].astype('category')
        if df_copy[c].isnull().any():
            df_copy[c] = df_copy[c].cat.add_categories(['MISSING'])
            df_copy[c] = df_copy[c].fillna('MISSING')
    return df_copy


def figsize_decorator(function):
    @wraps(function)
    def wrapper(*args, figsize=None, **kwargs):
        if figsize:
            plt.figure(figsize=figsize)
        return function(*args, **kwargs)

    return wrapper


def plot_in_grid(df,
                 columns,
                 plot_function,
                 plots_in_row,
                 y=None,
                 map_args=['value'],
                 size=4,
                 **kwargs):
    if columns is None:
        _columns = df.columns
    elif isinstance(columns, str):
        _columns = [columns]
    else:
        _columns = columns.copy()

    f = pd.melt(df, id_vars=[y] if y else None, value_vars=_columns)
    g = sns.FacetGrid(
        f,
        col="variable",
        col_wrap=plots_in_row,
        sharex=False,
        sharey=False,
        size=size)
    g = g.map(plot_function, *map_args, **kwargs)
    return g


def plot_distribution(df, columns=None, plots_in_row=5, size=4, **kwargs):
    '''
    plots quantitative features without NaN!

    df - DataFrame
    columns - columns or column if 'data' is DataFrame
    *args, **kwargs - arguments for sns.distplot

    plots distribution in grid with 'plots_in_row' columns.
    ---
    useful kwargs:
        fit - fit=scipy.stats.[johnsonsu/norm/lognorm]
                shows how the pdf fits distribution
    '''
    def my_distplot(x, **kwargs):
        sns.distplot(x, **kwargs)
        x = plt.xticks(rotation=90)

    return plot_in_grid(
        df, columns, my_distplot, plots_in_row, size=size, **kwargs)


def plot_histogram(df, columns=None, plots_in_row=5, size=4, **kwargs):
    '''
    plots ONLY quantitative or ONLY qualitative
    '''

    def my_hist(x, **kwargs):
        plt.hist(x, **kwargs)
        x = plt.xticks(rotation=90)

    return plot_in_grid(
        df, columns, my_hist, plots_in_row, size=size, **kwargs)


def plot_boxplot(df, columns, y, plots_in_row=3, size=4, **kwargs):
    '''
    df - DataFrame
    y - column, plots relative to it.
    columns - QUALITATIVE column names
    '''
    _kwargs = {}
    _kwargs.update(kwargs)

    df_filled = fill_with_missing(df, columns)

    def my_boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    return plot_in_grid(
        df_filled,
        columns,
        my_boxplot,
        plots_in_row,
        y=y,
        map_args=['value', y],
        size=size,
        **kwargs)


@figsize_decorator
def plot_pairplot(df, columns=None, *args, **kwargs):
    '''
    plots sns.pairplot()
    '''
    _kwargs = {
        'vars': columns,
        'kind': 'reg',
        'size': 4,
    }
    _kwargs.update(kwargs)

    return sns.pairplot(df, *args, **_kwargs)


@figsize_decorator
def plot_heatmap(df,
                 columns=None,
                 n_largest=None,
                 relative_to=None,
                 *args,
                 **kwargs):
    '''
    df - DataFrame
    columns - iterable, column names
    ----
    if 'n_largest' and 'relative_to' provided, plots heatmap with n_largest correlation relative to column:
    n_largest - int or None 
    relative_to - column name
    '''
    _kwargs = {}
    _kwargs.update(kwargs)
    if columns is not None:
        _columns = columns
    else:
        _columns = df.columns

    corrmat = df[_columns].corr()

    if any([n_largest, relative_to]):
        assert n_largest is not None \
            and relative_to is not None

    if n_largest and relative_to:
        cols = corrmat.nlargest(n_largest, relative_to)[relative_to].index
        corrmat = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(
            corrmat,
            cbar=True,
            annot=True,
            square=True,
            fmt='.2f',
            annot_kws={'size': 10},
            yticklabels=cols.values,
            xticklabels=cols.values)
        return hm
    else:
        return sns.heatmap(corrmat, *args, **kwargs)


@figsize_decorator
def plot_missing(df, absolute=True, *args, **kwargs):
    '''
    df - DataFrame
    absolute - bool, shows absolute number of missing values or relative (from 0 to 1)
    '''
    _kwargs = {}
    _kwargs.update(kwargs)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    if not absolute:
        missing = missing / df.shape[0]
    missing.plot.bar()
    missing
    return missing


def anova(frame, qualitative, relative_to):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls][relative_to].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


@figsize_decorator
def plot_anova(df, columns, relative_to):
    '''
    QUALITATIVE columns
    '''
    df_filled = fill_with_missing(df, columns)

    a = anova(df_filled, columns, relative_to)
    a['disparity'] = np.log(1. / a['pval'].values)
    sns.barplot(data=a, x='feature', y='disparity')
    x = plt.xticks(rotation=90)
    plt.show()


@figsize_decorator
def plot_correlations(df, columns, relative_to, corr_type='spearman'):
    '''
    QUANTITATIVE columns
    '''
    spr = pd.DataFrame()
    spr['feature'] = columns
    spr[corr_type] = [
        df[col].corr(df[relative_to], corr_type) for col in columns
    ]
    spr = spr.sort_values(corr_type)
    plt.figure(figsize=(6, 0.25 * len(columns)))
    sns.barplot(data=spr, y='feature', x=corr_type, orient='h')


def get_quanti_quali_columns(df, columns=None):
    '''
    returns tuple of two lists of column names with
    Quantitative and qualitive features.
    '''
    if columns is None:
        _columns = df.columns
    else:
        _columns = columns.copy()

    quantitative = [
        f for f in _columns if df.dtypes[f] != 'object'
    ]
    qualitative = [
        f for f in _columns if df.dtypes[f] == 'object'
    ]
    return quantitative, qualitative


def show_report_1(df_init, target_column, columns_init=None):
    df = df_init.copy()

    if not columns_init:
        columns = df.columns.copy()
    else:
        columns = columns.copy()

    print('COLUMN INFO:')
    print_column_info(df, columns)
    print()

    print('Missed values')
    plot_missing(df[columns])
    plt.show()

    quantitative, qualitative = get_quanti_quali_columns(df, columns)
    df_filled = fill_with_missing(df, qualitative)

    print('Quantitative correlations:')
    plot_correlations(
        df, quantitative, relative_to=target_column, corr_type='spearman')

    plt.show()
    # return df, qualitative, target_column
    print('Important qualitative Features (ANOVA):')
    plot_anova(df, qualitative, target_column)
    plt.show()

    plot_distribution(df, quantitative, fit=stats.norm)
    plot_histogram(df, qualitative)
    plot_boxplot(df_filled, qualitative, target_column)
    # plot_pairplot(df, columns)
    plot_heatmap(df, columns, n_largest=20,
                 relative_to=target_column, figsize=(20, 20))
    plt.show()
    return None
