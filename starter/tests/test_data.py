"""
Pytest module to test input data and modeling.

Author: Emily Travinsky
Date: 12/2023
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model


def test_nonexistent_column_raises_error(clean_data: pd.DataFrame,
                                         cat_features: list):
    """Function to test that a nonexistent column will raise a KeyError.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.
    """

    cat_features_invalid = cat_features.copy()
    cat_features_invalid.append("fake-column")

    train, _ = train_test_split(clean_data, test_size=0.20)

    with pytest.raises(KeyError):
        process_data(
            train,
            categorical_features=cat_features_invalid,
            label="salary",
            training=True
        )


def test_nonbinary_target_raises_error(clean_data: pd.DataFrame,
                                       cat_features: list):
    """Function to test that nonbinary target raises ValueError.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.
    """

    train, _ = train_test_split(clean_data, test_size=0.20)

    x_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    y_train[y_train == 1] = 4

    with pytest.raises(ValueError):
        train_model(x_train, y_train)


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
    """Function to test that expected columns are present in the dataframe.

    Args:
        trained_model (Pipeline): ML Pipeline with preprocessing and model.
    """

    assert isinstance(trained_model, Pipeline)


def test_predictions_type(preds: np.ndarray):
    """Function to test that predictions are returned as np.ndarray.

    Args:
        preds (np.ndarray): Numpy array of predictions for 'x_test' data.
    """

    assert isinstance(preds, np.ndarray)


def test_metrics_types(metrics: list):
    """Function to test that metrics are returned as a list of floats.

    Args:
        metrics (list): List of metrics [precision, recall, fbeta]
    """

    assert all(isinstance(x, float) for x in metrics)
