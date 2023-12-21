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
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference, compute_model_metrics


@pytest.fixture(scope='session', name='clean_data')
def fixture_clean_data() -> pd.DataFrame:
    """Fixture to import cleaned dataframe of census data.

    Returns:
        pd.DataFrame: Dataframe of census data with salary information.
    """

    df = pd.read_csv("./starter/data/census_cleaned.csv")

    return df


@pytest.fixture(scope='session', name='cat_features')
def fixture_cat_features() -> list:
    """Fixture to define categorical features in testing data.

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


@pytest.fixture(scope='session', name='encoder_lb')
def fixture_encoder_lb(clean_data: pd.DataFrame, cat_features: list) -> dict:
    """Fixture for data encoder and label binarizer.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.

    Returns:
        dict: Dictionary with data 'encoder' and 'lb' (label binarizer) keys.
    """

    # Process the training data with the process_data function.
    _, _, encoder, lb = process_data(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    return {'encoder': encoder, 'lb': lb}


@pytest.fixture(scope='session', name='trained_model')
def fixture_trained_model() -> Pipeline:
    """Fixture to load pretrained model for testing.

    Returns:
        Pipeline: ML pipeline with preprocessing and model.
    """

    model = joblib.load("./starter/tests/test_model.pkl")

    return model


@pytest.fixture(scope='session', name='test_data')
def fixture_test_data(clean_data: pd.DataFrame, cat_features: list,
                      encoder_lb: dict) -> dict:
    """Fixture to create dictionary of testing data for inference and metrics.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.
        encoder_lb (dict): Dictionary with keys for data 'encoder' and 'lb'.

    Returns:
        dict: Test data dictionary with keys for 'x_test' and 'y_test'.
    """

    x_test, y_test, _, _ = process_data(
        clean_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder_lb['encoder'],
        lb=encoder_lb['lb']
    )

    return {'x_test': x_test, 'y_test': y_test}


@pytest.fixture(scope='session', name='preds')
def fixture_preds(trained_model: Pipeline, test_data: dict) -> np.ndarray:
    """Fixture to run inference on testing data.

    Args:
        trained_model (Pipeline): Trained ML model to run inference.
        test_data (dict): Test data with dictionary keys for 'x_test' and
            'y_test'.

    Returns:
        np.ndarray: Numpy array of predictions for 'x_test' data.
    """

    preds = inference(trained_model, test_data['x_test'])

    return preds


@pytest.fixture(scope='session')
def metrics(test_data: dict, preds: np.ndarray) -> List[float]:
    """Fixture to calculate metrics for inference predictions on testing data.

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
