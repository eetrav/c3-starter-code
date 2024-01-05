"""
Python class to process training and testing data for salary prediction.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


class PreProcessor:
    def __init__(self,
                 df: pd.DataFrame,
                 categorical_features: list,
                 label: str = "",
                 training=True,
                 encoder=None,
                 lb=None):
        """Instantiate the PreProcessing class to store inputs.

        Args:
            df (pd.DataFrame): Dataframe containing the features and label. 
                Columns in `categorical_features`
            categorical_features (list): List containing the names of the 
                categorical features (default=[])
            label (str, optional): Name of the label column in `X`. If None,
                then an empty array will be returned for y (default=None).
                Defaults to "".
            training (bool, optional): Indicator if training mode or 
                inference/validation mode. Defaults to True.
            encoder ([type], optional): 
                sklearn.preprocessing._encoders.OneHotEncoder. Defaults to 
                None, only used if training=False.
            lb ([type], optional): sklearn.preprocessing._label.LabelBinarizer. 
                Defaults to None, only used if training=False.
        """

        self.df = df
        self.categorical_features = categorical_features
        self.label = label
        self.training = training
        self.encoder = encoder
        self.scaler = StandardScaler()
        self.lb = lb

        self.x_train = None
        self.x_test = None

    def train_test_split(self, test_size=0.2, stratify_by=None):
        """
        Run sklearn's train_test_split to create training and testing sets.

        Args:
            test_size (float, optional): Percentage of data to use for testing.
                Defaults to 0.2.
            stratify_by ([type], optional): Dataframe column to use for 
                stratification. Defaults to None.
        """
        self.x_train, self.x_test = train_test_split(self.df,
                                                     test_size=test_size,
                                                     stratify=stratify_by)

    def process_data(self, feature: str = "", value: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data used in the machine learning pipeline.

        Processes the data using one hot encoding for the categorical features
        and a label binarizer for the labels. This can be used in either training
        or inference/validation.

        Note: depending on the type of model used, you may want to add in
        functionality that scales the continuous data.


        Args:
            feature (str, optional): Optional feature to use for generating 
                testing data with only one feature-value pair; used to 
                calculate metrics on sliced data. Defaults to "".
            value (str, optional): Optional value of feature to use in data
                slicing. Both feature and value must be defined for data
                slicing.Defaults to "".

        Raises:
            ValueError: If column has a data type that is neither categorical
                nor numeric.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed input data, and processed
                labels if label exists.
        """

        if bool(feature) is not bool(value):
            print("Did you intend to slice the testing data? Both feature and \
                  value must be defined!")
            raise ValueError

        if self.training:
            x = self.x_train.copy()
        else:
            x = self.x_test.copy()
            # If we are slicing the data for performance metrics
            if feature and value:
                x = x[x[feature] == value]

        # Check if there were no instances of feature-value in the test set
        if x.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.label is not None:
            y = x[self.label]
            x = x.drop([self.label], axis=1)
        else:
            y = np.array([])

        x_categorical = x[self.categorical_features].values
        x_continuous = x.drop(*[self.categorical_features], axis=1)

        for col in x_continuous.columns:
            if not is_numeric_dtype(x_continuous[col]):
                print(col, "was not listed as categorical but is also not numeric!")
                raise ValueError

        if self.training is True:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.lb = LabelBinarizer()
            x_categorical = self.encoder.fit_transform(x_categorical)
            x_continuous = self.scaler.fit_transform(x_continuous)
            y = self.lb.fit_transform(y.values).ravel()
        else:
            x_categorical = self.encoder.transform(x_categorical)
            x_continuous = self.scaler.transform(x_continuous)
            try:
                y = self.lb.transform(y.values).ravel()
            # Catch the case where y is None because we're doing inference.
            except AttributeError:
                pass

        x = np.concatenate([x_continuous, x_categorical], axis=1)
        return x, y
