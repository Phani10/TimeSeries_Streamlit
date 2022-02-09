from auto_config import config
from generic_functions import mape

import pandas as pd
from xgboost import XGB
import streamlit as st

def train():
    df = pd.read_pickle(config['save_path'] + '.pkl')

    gb_reg = xgb.XGBRegressor(max_depth=5, n_estimators=100, n_jobs=2,
                                   objectvie='reg:squarederror', booster='gbtree',
                                   random_state=42, learning_rate=0.03)

    st.write(gb_reg)

    x = pd.date_range(start=df['date'].min(), end=df['date'].max())
    cut_off_date = x[round(len(x)*0.9)]

    train=df[(df['date'] >= x.min()) & (df['date'] <= cut_off_date)]
    test=df[(df['date'] > cut_off_date)]

    train.drop(['date'],axis=1, inplace=True)
    test.drop(['date'],axis=1, inplace=True)

    train_y=train['count']
    train_x=train[['working_day', 'public_holiday']]
    test_y=test['count']
    test_x=test[['working_day', 'public_holiday']]

    gb_reg.fit(train_x,train_y)

    y_pred = gb_reg.predict(train_x) # Predictions

    st.write(mape(y_pred,train_y))

def app():
    train()
