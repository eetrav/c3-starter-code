"""
Pytest module to test input data and modeling.

Author: Emily Travinsky
Date: 12/2023
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.data import PreProcessor
from starter.starter.ml.model import train_model


def test_nonexistent_column_raises_error(clean_data: pd.DataFrame,
                                         cat_features: list):
    """
    Function to test that a nonexistent column will raise a KeyError.

    Args:
        clean_data (pd.DataFrame): Cleaned dataframe with features and target.
        cat_features (list): List of categorical features in dataframe.
    """

    cat_features_invalid = cat_features.copy()
    cat_features_invalid.append("fake-column")

    preprocessor = PreProcessor(
        clean_data,
        categorical_features=cat_features_invalid,
        label="salary",
        training=True
    )

    preprocessor.train_test_split(
        test_size=0.20, stratify_by=clean_data["sex"]
    )

    with pytest.raises(KeyError):
        preprocessor.process_data()


def test_nonbinary_target_raises_error(preprocessor: PreProcessor):
    """
    Function to test that nonbinary target raises ValueError.

    Args:
        preprocessor (PreProcessor): Instantiated Preprocessor with data 
            split into train and test sets.
    """

    # Process the training data to generate encoder and label_binarizer
    x_train, y_train = preprocessor.process_data()

    y_train[y_train == 1] = 4

    with pytest.raises(ValueError):
        train_model(x_train, y_train)


def check_error_for_missing_cat_column(preprocessor: PreProcessor):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        preprocessor (PreProcessor): Instantiated Preprocessor with data 
            split into train and test sets.
    """

    # Remove one categorical variable from the categorical features list
    preprocessor.categorical_features.remove("salary")

    with pytest.raises(ValueError):
        preprocessor.process_data()


def test_model_type(trained_model: RandomForestClassifier):
    """
    Function to test that the 

    Args:
        trained_model (RandomForestClassifier): Trained Random Forest model.
    """

    assert isinstance(trained_model, RandomForestClassifier)


def test_predictions_type(preds: np.ndarray):
    """
    Function to test that predictions are returned as np.ndarray.

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
