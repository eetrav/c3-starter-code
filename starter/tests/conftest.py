import pandas as pd
import pytest


@pytest.fixture(scope='session')
def messy_data():

    df = pd.read_csv("messy_data_sample.csv")

    return df


@pytest.fixture(scope='session')
def clean_data():

    df = pd.read_csv("clean_data_sample.csv")

    return df
