import pandas as pd
from pandas.api.types import is_string_dtype
import pytest
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data


def test_nonexistent_column(clean_data: pd.DataFrame):
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
        "fake-feature"
    ]

    train, _ = train_test_split(clean_data, test_size=0.20)

    with pytest.raises(KeyError):
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )


def test_binary_target(clean_data: pd.DataFrame):
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
        "native-country"
    ]

    train, _ = train_test_split(clean_data, test_size=0.20)

    with pytest.raises(KeyError):
        _, y_train, _, _ = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )

    # https://stackoverflow.com/questions/40595967/
    assert (((y_train == 0) | (y_train == 1)).all())
