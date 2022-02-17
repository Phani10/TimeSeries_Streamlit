from config import config

import pandas as pd


def app(df):
    id_cols = list()
    cat_cols = list()
    num_cols = list()
    ts_cols = list()

    if config['auto_identify_data_types'] == 'y':

        for col in df.columns:
            print(col)
            if df[col].isnull().sum() > df.shape[0]*0.8:
                cat_cols.append(col)
                continue

            if (df[col].dtype == 'datetime64') :
                ts_cols.append(col)

            elif (df[col].nunique() > df.shape[0]*0.8) & (df[col].dtype != 'float64'):
                id_cols.append(col)

            elif df[col].nunique() < 20:
                cat_cols.append(col)

            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    ts_cols.append(col)
                    continue
                except ValueError:
                    cat_cols.append(col)

            elif (df[col].dtype == 'int64') | (df[col].dtype == 'float64'):
                num_cols.append(col)

            else:
                cat_cols.append(col)

        config['cat_cols'] = cat_cols
        config['num_cols'] = num_cols
        config['id_cols'] = id_cols
        config['ts_cols'] = ts_cols

        time_col = ''
        if len(ts_cols) > 0:
            config['time_col'] = ts_cols[0]
        else:
            config['time_col'] = ''

        print('cat_cols: ', cat_cols)
        print('num_cols: ', cat_cols)
        print('id_cols: ', cat_cols)
        print('ts_cols: ', cat_cols)

    else:
        config['cat_cols'] = config['cat_cols'].replace(' ','').split(',')
        config['num_cols'] = config['num_cols'].replace(' ','').split(',')
        config['id_cols'] = config['id_cols'].replace(' ','').split(',')
        config['ts_cols'] = config['ts_cols'].replace(' ','').split(',')

    return config