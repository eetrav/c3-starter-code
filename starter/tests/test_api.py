"""
Pytest module to test input inference API.

Author: Emily Travinsky
Date: 01/2024
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.data import PreProcessor
from starter.starter.ml.model import train_model

from starter.main import Person


def test_model_predicts_over_50k(person: Person):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        preprocessor (PreProcessor): Trained data Preprocessor.
    """

    # Test
    pass
