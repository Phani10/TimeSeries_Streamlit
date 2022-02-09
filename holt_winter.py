from config import config
from generic_functions import mape

import pandas as pd
import streamlit as st
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
import matplotlib.pyplot as plt


def build_model(df, path):
    series = df[config['target_col']]
    n = len(series)
    train = series[:round(n*0.9)]

    model_hw = HWES(train, seasonal_periods=7, trend='add', seasonal='add')
    hw_model_result = model_hw.fit(optimized=True, use_brute=True)

    file = open(path + 'holt_winters_model.pkl', 'wb')
    pickle.dump(hw_model_result, file)
    file.close()
    st.success('Model built and saved successfully')

    st.write(hw_model_result.summary())
    return hw_model_result


def load_model(path):
    model = pickle.load(open(path + 'holt_winters_model.pkl', 'rb'))
    st.success('Model loaded successfully')
    st.write(model.summary())
    return model


def hw_model(df, path):
    st.subheader('Do you want to re-train the model or load the existing model?')
    if st.button('Re-build model'):
        model = build_model(df, path)
        return model

    elif st.button('Load existing model'):
        model = load_model(path)
        return model


def prediction(df, model):
    series = df[config['target_col']]
    n = len(series)
    train = series[:round(n*0.9)]
    test = series[round(n*0.9):]

    # Obtain predicted values for train
    st.subheader("Prediction on Train dataset")
    end = len(train) - 1
    predictions_train = model.predict(start=0, end=end).rename('Model Predictions')
    # Plot predictions against known values
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Test')
    ax.plot(predictions_train, label="predictions")
    ax.plot(train, label="actual")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Prediction on Test dataset")
    # Obtain predicted values test
    predictions = model.forecast(steps=len(test))

    fig, ax = plt.subplots(figsize=(15,10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Test')
    ax.plot(predictions, label="predictions")
    ax.plot(test, label="actual")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Model MAPE")
    st.write('The MAPE for this model on test data is {} %'.format(round(mape(test, predictions),2)))


def app():
    df = pd.read_pickle(config['save_path'] + '.pkl')
    model = hw_model(df, config['model_path'])
    if model:
        prediction(df, model)
