import joblib

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from starter.starter.ml.model import compute_model_metrics

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model


def test_nonexistent_column_raises_KeyError(clean_data: pd.DataFrame,
                                            cat_features: list):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "fake-column"
    ]

    train, _ = train_test_split(clean_data, test_size=0.20)

    with pytest.raises(KeyError):
        process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )


def test_nonbinary_target_raises_ValueError(clean_data: pd.DataFrame,
                                            cat_features: list):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    train, _ = train_test_split(clean_data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    y_train[y_train == 1] = 4

    with pytest.raises(ValueError):
        train_model(X_train, y_train)


def check_cat_columns(clean_data: pd.DataFrame, cat_features: list):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    train, _ = train_test_split(clean_data, test_size=0.20)

    with pytest.raises(ValueError):
        process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )


def test_model_type(trained_model: Pipeline):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    assert isinstance(trained_model, Pipeline)


def test_predictions_type(predictions: np.ndarray):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    assert isinstance(predictions, np.ndarray)
