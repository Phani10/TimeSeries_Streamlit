from config import config
from load_data import main

import streamlit as st
import pandas as pd


def display_data():
    st.header("Welcome to Time Series analysis")
    st.write("Click the below button to upload the data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully. Shape of the file is {}".format(df.shape))
        st.dataframe(df.head())
        main()


def app():
    display_data()
