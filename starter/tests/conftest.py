import joblib
import pandas as pd
import pytest

from starter.starter.ml.data import process_data


@pytest.fixture(scope='session')
def clean_data():

    df = pd.read_csv("./starter/tests/clean_data_sample.csv")

    return df


@pytest.fixture(scope='session', name='cat_features')
def fixture_cat_features():

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    return cat_features


@pytest.fixture(scope='session')
def encoder_lb(cat_features: list):
    # Process the training data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return {'encoder': encoder, 'lb': lb}


@pytest.fixture(scope='session')
def trained_model():

    model = joblib.load("./starter/tests/test_model.pkl")

    return model
