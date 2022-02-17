from auto_config import config
from generic_functions import mape
from generic_functions import get_best_params

import pandas as pd
import lightgbm as lgb
import streamlit as st
import pickle
import matplotlib.pyplot as plt


def parameter_tuning(train_x, train_y, test_x, test_y):
    print("Light GBM parameter tuning")

    n_estimators = [100, 200, 500]
    learning_rate = [0.1, 0.01, 0.001]
    num_leaves = [15, 30, 40, 50]
    boosting_type = ['dart', 'gbdt']

    my_list = list()
    for b in boosting_type:
        for e in n_estimators:
            for lr in learning_rate:
                for n in num_leaves:

                    lgb_model = lgb.LGBMRegressor(boosting_type=b, num_leaves=n, max_depth=-1,
                                                   learning_rate=lr, n_estimators=e, importance_type='split')
                    lgb_model.fit(train_x, train_y)
                    lgb_pred = lgb_model.predict(test_x)
                    score = mape(test_y, lgb_pred)

                    print("Boosing type: {}, n_estimators: {}, learning rate: {}, \
                                  Num leaves: {}, mape: {}".format(b, e, lr, n, score))

                    my_list.append([b, e, lr, n, score])

    params = get_best_params(my_list)
    return params


def build_model(df, path):
    x = pd.date_range(start=df['date'].min(), end=df['date'].max())
    cut_off_date = x[round(len(x) * 0.9)]
    train = df[(df['date'] >= x.min()) & (df['date'] <= cut_off_date)]
    test=df[(df['date'] > cut_off_date)]

    train.drop(['date'],axis=1, inplace=True)
    test.drop(['date'],axis=1, inplace=True)

    train_y = train['count']
    train_x = train.drop('count', axis=1)
    test_y = test['count']
    test_x = test.drop('count', axis=1)

    params = parameter_tuning(train_x, train_y, test_x, test_y)

    model = lgb.LGBMRegressor(boosting_type=params[0], num_leaves=params[3], max_depth=-1,
                              learning_rate=params[2], n_estimators=params[1], importance_type='split')
    model.fit(train_x, train_y)

    file = open(path + 'lgb_model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()
    st.success('Model built and saved successfully')

    return model


def load_model(path):
    model = pickle.load(open(path + 'lgb_model.pkl', 'rb'))
    st.success('Model loaded successfully')
    return model


def lgb_model(df, path):
    st.subheader('Do you want to re-train the model or load the existing model?')
    if st.button('Re-build model'):
        model = build_model(df, path)
        return model

    elif st.button('Load existing model'):
        model = load_model(path)
        return model


def prediction(df, model):
    x = pd.date_range(start=df['date'].min(), end=df['date'].max())
    cut_off_date = x[round(len(x)*0.9)]
    train=df[(df['date'] >= x.min()) & (df['date'] <= cut_off_date)]
    test=df[(df['date'] > cut_off_date)]

    train.drop(['date'],axis=1, inplace=True)
    test.drop(['date'],axis=1, inplace=True)

    train_y=train['count']
    train_x=train.drop('count', axis=1)
    test_y=test['count']
    test_x=test.drop('count', axis=1)


    # Obtain predicted values for train
    st.subheader("Prediction on Train dataset")
    end = len(train) - 1
    predictions_train = model.predict(train_x)
    # Plot predictions against known values
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Train')
    ax.plot(predictions_train, label="predictions")
    ax.plot(list(train_y), label="actual")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Prediction on Test dataset")
    # Obtain predicted values test
    predictions = model.predict(test_x)

    fig, ax = plt.subplots(figsize=(15,10))
    plt.xlabel('Date')
    plt.ylabel(config['target_col'])
    plt.title('Actual vs. Predicted - Test')
    ax.plot(predictions, label="predictions")
    ax.plot(list(test_y), label="actual")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Model MAPE")
    mape_val = round(mape(test_y, predictions),2)
    st.write('The MAPE for this model on test data is {} %'.format(mape_val))

    path = config['model_path']
    results_dict = pickle.load(open(path + 'results_dict.pkl', 'rb'))
    results_dict['Light GBM Regression Model'] = str(mape_val)
    file = open(path + 'results_dict.pkl', 'wb')
    pickle.dump(results_dict, file)
    file.close()
    st.success('Results saved')


def app():
    df = pd.read_pickle(config['save_path'] + '.pkl')
    model = lgb_model(df, config['model_path'])
    if model:
        prediction(df, model)
