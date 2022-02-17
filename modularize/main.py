import home_page
import eda
import arima
import holt_winter
import ses
import fb_prophet
import model_results_summary
import xgboost_model
import lgbm_model
import fb_prophet_multivariable

import streamlit as st

st.set_page_config(layout='wide')
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

PAGES = {
    "Home Page": home_page,
    "Model Results Summary": model_results_summary,
    "EDA - Exploratory Data Analysis": eda,
    "ARIMA Model": arima,
    "Simple Exponential Smoothing": ses,
    "Holt-Winters Exponential Smoothing": holt_winter,
    "FB Prophet - Univariable": fb_prophet,
    "XGBoost Regression Model": xgboost_model,
    "Light GBM Regression Model": lgbm_model,
    "FB Prophet - Multivariable": fb_prophet_multivariable
}

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][area-expanded='true"] > div:first-child {
        width: 240px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Time Series analysis')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
