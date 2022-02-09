import streamlit as st
import home_page, eda, arima, holt_winter, ses, fb_prophet, playground, xgboost

st.set_page_config(layout='wide')
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

PAGES = {
    "Home": home_page,
    "Results Summary": playground,
    "EDA - Exploratory Data Analysis": eda,
    "ARIMA Model": arima,
    "Simple Exponential Smoothing": ses,
    "Holt-Winters Exponential Smoothing": holt_winter,
    "FB Prophet": fb_prophet,
    "XGBoost": xgboost
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
