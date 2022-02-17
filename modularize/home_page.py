from config import config
from load_data import main

import streamlit as st
import pandas as pd
import pickle
import yaml
import os


def display_data():
    st.title("Welcome to Time Series analysis")
    st.write("Click the below button to upload the data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully. Shape of the file is {}".format(df.shape))
        st.dataframe(df.head())

        st.subheader("Please select target variable")
        target = st.radio("", list(df.columns))

        st.subheader("Please select date column")
        date = st.radio(" ", list(df.columns))

        submit = st.button('Submit')
        if submit:
            results_dict = {
                "ARIMA Model": '',
                "Simple Exponential Smoothing": '',
                "Holt-Winters Exponential Smoothing": '',
                "FB Prophet - Univariable": '',
                "XGBoost Regression Model": '',
                "Light GBM Regression Model": '',
                "FB Prophet - Multivariable": ''
            }
            config['target_col'] = target
            config['time_col'] = date

            with open(f'{os.path.dirname(os.path.abspath(__file__))}/config.yml', 'w') as f:
                yaml.dump(config, f)

            file = open(config['model_path'] + 'results_dict.pkl', 'wb')
            pickle.dump(results_dict, file)
            file.close()

            df.rename(columns={date:'date'}, inplace=True)
            main(df)
            st.success('Target and date columns saved successfully')

def app():
    display_data()
