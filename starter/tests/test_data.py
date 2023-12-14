import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data


def test_target_column(messy_data: pd.DataFrame):
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

    train, test = train_test_split(messy_data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
