"""
Pytest configuration to set-up reusable variables for testing.

Author: Emily Travinsky
Date: 12/2023
"""

from typing import List

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.data import PreProcessor
from starter.starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope='session', name='clean_data')
def fixture_clean_data() -> pd.DataFrame:
    """
    Fixture to import a sample of cleaned census data.

    Returns:
        pd.DataFrame: Dataframe of census data with salary information.
    """

    df = pd.read_csv("./starter/tests/clean_data_sample.csv")

    return df


@pytest.fixture(scope='session', name='cat_features')
def fixture_cat_features() -> list:
    """
    Fixture to define categorical features in testing data.

    Returns:
        list: List of categorical features in cleaned dataframe.
    """

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


@pytest.fixture(scope='session', name='encoder')
def fixture_encoder() -> OneHotEncoder:
    """
    Fixture to import a pre-trained categorical encoder for testing.

    Returns:
        OneHotEncoder: Pre-trained one-hot encoder for testing.
    """

    encoder = joblib.load("./starter/tests/encoder.pkl")

    return encoder


@pytest.fixture(scope='session', name='preprocessor')
def fixture_preprocessor(clean_data: pd.DataFrame, cat_features: list) -> PreProcessor:
    """
    Fixture to instantiate PreProcessor for model training and testing.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.

    Returns:
        PreProcessor: Instantiated PreProcessor to use with model training.
    """

    # Process the training data with the process_data function.
    preprocessor = PreProcessor(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    preprocessor.train_test_split(
        test_size=0.20, stratify_by=clean_data["sex"]
    )

    return preprocessor


@pytest.fixture(scope='session', name='trained_model')
def fixture_trained_model(preprocessor: PreProcessor) -> RandomForestClassifier:
    """
    Fixture to train model and test output type.

    Args:
        preprocessor (PreProcessor): Instantiated PreProcessor to use with
            model training.

    Returns:
        RandomForestClassifier: RandomForestClassifier trained with mini- clean
            dataset.
    """

    X_train, y_train = preprocessor.process_data()

    # Process the test data with the process_data function.
    preprocessor.training = False
    X_test, y_test = preprocessor.process_data()

    # Train and save the model
    model = train_model(X_train, y_train)

    return model


@pytest.fixture(scope='session', name='testing_model')
def fixture_testing_model() -> RandomForestClassifier:
    """
    Fixture to load pretrained model for testing.

    Returns:
        RandomForestClassifier: Classifier pretrained with all clean data.
    """

    testing_model = joblib.load("./starter/tests/test_model.pkl")

    return testing_model


@pytest.fixture(scope='session', name='test_data')
def fixture_test_data(preprocessor: PreProcessor, encoder: OneHotEncoder) -> dict:
    """
    Fixture to create dictionary of testing data for inference and metrics.

    Args:
        preprocessor (PreProcessor): Instantiated PreProcessor to use with
            model training.
        encoder (OneHotEncoder): Pre-trained one-hot encoder for testing. Used
            to replace the preprocessor encoder, insuring that input data will
            encode in the same way the pre-trained model saw during training.

    Returns:
        dict: Test data dictionary with keys for 'x_test' and 'y_test'.
    """

    preprocessor.training = False
    preprocessor.encoder = encoder
    x_test, y_test = preprocessor.process_data()

    return {'x_test': x_test, 'y_test': y_test}


@pytest.fixture(scope='session', name='preds')
def fixture_preds(testing_model: RandomForestClassifier, test_data: dict) -> np.ndarray:
    """
    Fixture to run inference on testing data.

    Args:
        trained_model (RandomForestClassifier): Trained ML model to run
            inference.
        test_data (dict): Test data with dictionary keys for 'x_test' and
            'y_test'.

    Returns:
        np.ndarray: Numpy array of predictions for 'x_test' data.
    """

    preds = inference(testing_model, test_data['x_test'])

    return preds


@pytest.fixture(scope='session')
def metrics(test_data: dict, preds: np.ndarray) -> List[float]:
    """
    Fixture to calculate metrics for inference predictions on testing data.

    Args:
        test_data (dict): Test data with dictionary keys for 'x_test' and
            'y_test'.
        preds (np.ndarray): Numpy array of predictions for 'x_test' data.

    Returns:
        List[float]: List of metrics [precision, recall, fbeta]
    """

    precision, recall, fbeta = compute_model_metrics(test_data['y_test'],
                                                     preds)

    return [precision, recall, fbeta]


@pytest.fixture(scope="session")
def salary_over_50k():
    """AI is creating summary for salary_over_50k

    Returns:
        [type]: [description]
    """

    features = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capitol_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }

    return features


@pytest.fixture(scope="session")
def salary_under_50k():

    features = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capitol_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    return features
