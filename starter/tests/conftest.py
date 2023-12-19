import joblib
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference, compute_model_metrics


@pytest.fixture(scope='session', name='clean_data')
def fixture_clean_data():

    df = pd.read_csv("./starter/data/census_cleaned.csv")

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


@pytest.fixture(scope='session', name='encoder_lb')
def fixture_encoder_lb(clean_data: pd.DataFrame, cat_features: list):
    # Process the training data with the process_data function.
    _, _, encoder, lb = process_data(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return {'encoder': encoder, 'lb': lb}


@pytest.fixture(scope='session', name='trained_model')
def fixture_trained_model():

    model = joblib.load("./starter/tests/test_model.pkl")

    return model


@pytest.fixture(scope='session', name='test_data')
def fixture_test_data(clean_data: pd.DataFrame, cat_features: list,
                      trained_model: Pipeline, encoder_lb: dict):

    X_test, y_test, _, _ = process_data(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder_lb['encoder'],
        lb=encoder_lb['lb']
    )

    return {'X_test': X_test, 'y_test': y_test}


@pytest.fixture(scope='session', name='preds')
def fixture_preds(trained_model: Pipeline, test_data: dict):

    preds = inference(trained_model, test_data['X_test'])

    return preds


@pytest.fixture(scope='session')
def metrics(test_data: dict, preds: np.ndarray):

    precision, recall, fbeta = compute_model_metrics(test_data['y_test'],
                                                     preds)

    return [precision, recall, fbeta]
