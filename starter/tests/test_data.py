import pandas as pd
from pandas.api.types import is_string_dtype


def test_column_format(messy_data: pd.DataFrame):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    for col in messy_data.columns:
        assert not (" " in col)
