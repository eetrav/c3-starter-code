import joblib
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def clean_data():

    df = pd.read_csv("./starter/tests/clean_data_sample.csv")

    return df


@pytest.fixture(scope='session')
def trained_model():

    model = joblib.load("./starter/tests/test_model.pkl")

    return model
