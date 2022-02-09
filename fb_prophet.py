from config import config
from generic_functions import mape

import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from fbprophet import Prophet


def build_model(df, path):
    df = df[["date", "count"]]
    df = df.rename({'date': 'ds', 'count': 'y'}, axis='columns')

    x = pd.date_range(start=df['date'].min(), end=df['date'].max())
    cut_off_date = x[round(len(x) * 0.9)]

    train = df[(df['date'] >= x.min()) & (df['date'] <= cut_off_date)]

    model = Prophet(interval_width=0.95, daily_seasonality=False)
    model.fit(train)

    file = open(path + 'fb_prophet_model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()
    st.success('Model built and saved successfully')

    return model


def load_model(path):
    model = pickle.load(open(path + 'fb_prophet_model.pkl', 'rb'))
    st.success('Model loaded successfully')
    return model


def prophet_model(df, path):
    st.subheader('Do you want to re-train the model or load the existing model?')
    if st.button('Re-build model'):
        model = build_model(df, path)
        return model

    elif st.button('Load existing model'):
        model = load_model(path)
        return model


def prediction(df, model):
    df = df[["date", "count"]]
    df = df.rename({'date': 'ds', 'count': 'y'}, axis='columns')

    x = pd.date_range(start=df['date'].min(), end=df['date'].max())
    cut_off_date = x[round(len(x) * 0.9)]

    train = df[(df['date'] >= x.min()) & (df['date'] <= cut_off_date)].copy()
    test = df[(df['date'] > cut_off_date)].copy()

    # Obtain predicted values for train
    predictions_train = model.predict(train)

    train_preds_df = pd.concat([train.set_index('ds')['y'],
                                predictions_train.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]],axis=1)

    st.subheader('Model Summary')
    fig1 = model.plot(predictions_train)
    st.pyplot(fig1)

    fig2 = model.plot_components(predictions_train)
    st.pyplot(fig2)

    st.subheader("Prediction on Train dataset")
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Test')
    ax.plot(train_preds_df['y'], label="predictions")
    ax.plot(train_preds_df['yhat'], label="actual")
    plt.legend()
    st.pyplot(fig)


    ######################################## Test ##########################################

    # Obtain predicted values for train
    st.subheader("Prediction on Test dataset")
    predictions_test = model.predict(test)
    test_preds_df = pd.concat([test.set_index('ds')['y'],
                               predictions_test.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]],axis=1)
    st.write('Sample Test predictions')
    st.dataframe(test_preds_df.head())

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Test')
    ax.plot(test_preds_df['y'], label="predictions")
    ax.plot(test_preds_df['yhat'], label="actual")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Model MAPE")
    st.write('The MAPE for this model on test data is {} %'.format(round(mape(test_preds_df['y'],
                                                                              test_preds_df['yhat']),2)))


def app():
    df = pd.read_pickle(config['save_path'] + '.pkl')
    model = prophet_model(df, config['model_path'])
    if model:
        prediction(df, model)
