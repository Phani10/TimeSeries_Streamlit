from config import config
from generic_functions import mape

import pandas as pd
import streamlit as st
import pickle
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_model(df, path):
    series = df[config['target_col']]
    model = auto_arima(series, start_p=0, start_q=0,
                       max_p=2, max_q=2, m=5, start_P=0, start_Q=0,
                       max_P=2, max_Q=2,
                       seasonal=True,
                       d=0, D=0, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    n = len(series)
    train = series[:round(n*0.9)]

    sarimax_model = SARIMAX(train, order=model.get_params()['order'],
                            seasonal_order=model.get_params()['seasonal_order'])
    sarimax_model_result = sarimax_model.fit()

    file = open(path + 'arima_model.pkl', 'wb')
    pickle.dump(sarimax_model_result, file)
    file.close()
    st.success('Model built and saved successfully')

    st.write(sarimax_model_result.summary())
    return sarimax_model_result


def load_model(path):
    model = pickle.load(open(path + 'arima_model.pkl', 'rb'))
    st.success('Model loaded successfully')
    st.write(model.summary())
    return model


def model_results(model):
    st.subheader('Model Summary')
    plots = model.plot_diagnostics(figsize=(15, 12))
    st.pyplot(plots)


def arima_model(df, path):
    st.subheader('Do you want to re-train the model or load the existing model?')
    if st.button('Re-build model'):
        model = build_model(df, path)
        model_results(model)
        return model

    elif st.button('Load existing model'):
        model = load_model(path)
        model_results(model)
        return model


def prediction(df, model):
    series = df[config['target_col']]
    n = len(series)
    train = series[:round(n*0.9)]
    test = series[round(n*0.9):]

    # Obtain predicted values for train
    st.subheader("Prediction on Train dataset")
    end = len(train) - 1
    predictions_train = model.predict(start=0, end=end, dynamic=False, typ='levels').rename(
        'Model Predictions')
    # Plot predictions against known values
    import matplotlib.pyplot as plt
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
    start = len(train)
    end = len(train) + len(test) - 1
    predictions = model.predict(start=start, end=end, dynamic=False, typ='levels').rename(
        'Predictions on Test data')
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
    model = arima_model(df, config['model_path'])
    if model:
        prediction(df, model)

