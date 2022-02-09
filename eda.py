from config import config

import plotly.express as px
import pandas as pd
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def sample_records(df):
    st.subheader("First 5 records")
    st.dataframe(df.head(5))

    st.subheader("Last 5 records")
    st.dataframe(df.tail(5))

    st.subheader("Sample records")
    st.dataframe(df.sample(5))


def missing_values(df):
    st.subheader('Missing Values')
    x = pd.date_range(start=df['date'].min(), end=df['date'].max()).difference(df['date'])
    if len(x) > 0:
        st.write("There are values missing for {} days".format(len(x)))
        if len(x) > 5:
            st.write("Sample missing dates are ", x[:5])
        else:
            st.write("Missing dates are: ", x)
    else:
        st.write("There are no missing values in this data".format(len(x)))


def describe(df):
    st.subheader('Data Overview')
    st.dataframe(df.describe())


def plotly_plot(df):
    st.subheader("Time series visualization")
    fig = px.line(df, x='date', y=config['target_col'], title=config['target_col'])

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(fig)


def seasonal_decomposing(df):
    st.subheader('Seasonal Decomposing')
    series = df[config['target_col']]
    st.write('Additive model')
    result = seasonal_decompose(series, model='additive', period=7)
    st.caption("Original values")
    st.line_chart(result.observed)
    st.caption("Trend")
    st.line_chart(result.trend)
    st.caption("Seasonal")
    st.line_chart(result.seasonal)
    st.caption("Residual")
    st.line_chart(result.resid)

    st.write('Multiplicative model')
    result = seasonal_decompose(series, model='Multiplicative', period=7)
    st.caption("Original values")
    st.line_chart(result.observed)
    st.caption("Trend")
    st.line_chart(result.trend)
    st.caption("Seasonal")
    st.line_chart(result.seasonal)
    st.caption("Residual")
    st.line_chart(result.resid)


def acf_pacf_plots(df):
    st.subheader('Auto Correlation Graphs')
    acf_plot = plot_acf(df[config['target_col']], lags=30, title='Auto Correlation Function (ACF)')
    st.pyplot(acf_plot)
    st.subheader('PACF Plot')
    pacf_plot = plot_pacf(df[config['target_col']], lags=30, title='Partial Auto Correlation Function (PACF)')
    st.pyplot(pacf_plot)


def adf_test(df):
    st.subheader('ADF test for Stationarity check')
    # ADF test for checking stationary
    series = df.loc[:, config['target_col']].values
    result = adfuller(series, autolag='AIC')
    st.write(f'ADF Statistic: {round(result[0], 2)}')
    st.write(f'p-value: {round(result[1], 5)}')
    for key, value in result[4].items():
        st.write('Critial Values: {}, {}'.format(key, round(value, 2)))


def app():
    st.header('Data analysis')
    df = pd.read_pickle(config['save_path'] + '.pkl')
    sample_records(df)
    describe(df)
    # missing_values(df)
    # plotly_plot(df)
    seasonal_decomposing(df)
    # acf_pacf_plots(df)
    # adf_test(df)
