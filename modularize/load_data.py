from auto_config import app

import pandas as pd
from numpy import where
import pickle


def load_data_customized(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    print("Data loaded successfully from {}".format(path))

    grouped_df_1 = df.groupby(['year', 'month', 'day'])['count'].sum().reset_index()
    grouped_df_2 = df.groupby(['year', 'month', 'day'])[['working_day', 'weekend_day',
                                                         'public_holiday']].max().reset_index()

    merged_df = pd.merge(grouped_df_1, grouped_df_2, how='left', on=['year', 'month', 'day'])
    merged_df['date'] = pd.to_datetime(merged_df[['year', 'month', 'day']], format='%Y%m%d')
    merged_df.drop(['year', 'month', 'day'], axis='columns', inplace=True)

    return merged_df


def missing_values(df, config):
    for i in config['cat_cols']:
        df[i] = df[i].fillna(df[i].mode()[0])

    for i in config['num_cols']:
        df[i] = df[i].fillna(df[i].median())

    return df


def iqr_outlier(df, config):
    cols = list(set(config['num_cols']) - set(config['target_col']))

    for i in cols:
        series = df[i]

        lower_hinge = series.quantile(0.25)
        upper_hinge = series.quantile(0.75)
        iqr = upper_hinge - lower_hinge

        upper_limit = upper_hinge + 1.5 * iqr
        lower_limit = lower_hinge - 1.5 * iqr

        series = where(series > upper_limit, upper_limit, series)
        series = where((series < lower_limit) & (series > series.min()), lower_limit, series)

        df[i] = series

    print("Outlier treatment done")
    return df


def load_data(df, config):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    df = missing_values(df, config)
    df = iqr_outlier(df, config)

    return df


def save_data(df, path):
    file = open(path + '.pkl', 'wb')
    pickle.dump(df, file)
    file.close()
    df.to_csv(path + '.csv', index=None)
    print("Dataframe saved successfully at {}".format(path))


def main(df):
    config = app(df)
    print(config)
    data = load_data(df, config)
    save_data(data, config['save_path'])
    return data
