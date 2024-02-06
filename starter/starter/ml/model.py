"""
Python module with functionality to train and test ML models to predict salary.

Author: Emily Travinsky
Date: 12/2023
"""

from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from starter.starter.ml.data import PreProcessor

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import fbeta_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.


def train_model(x_train: np.ndarray,
                y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a Random Forest model and returns it.

    Constructs a model pipeline to:
    1) Performs feature pre-processing to normalize and zero-center
    non-categorical variables
    2) Train a Random Forest model with cross-fold validation.

    Inputs
    ------
    x_train (np.ndarray) : np.array
        Training data.
    y_train (np.ndarray) : np.array
        Labels.

    Returns
    -------
    model (RandomForestClassifier) :
        Trained machine learning model.
    """

    # Check that target values are binary
    # https://stackoverflow.com/questions/40595967/
    if ~((y_train == 0) | (y_train == 1)).all():
        raise ValueError

    # From scikit-learn LogisticRegressionCV documentation:
    # ‘newton-cholesky’ is a good choice for n_samples >> n_features,
    # especially with one-hot encoded categorical features with rare
    # categories.
    # model = LogisticRegressionCV(solver="lbfgs", cv=5)

    # model.fit(x_train, y_train)

    clf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [45, 60, 75],
        'max_depth': [15, 25, 35]
    }

    grid_clf = GridSearchCV(clf, param_grid, cv=5)
    grid_clf.fit(x_train, y_train)

    print("Best Random Forest training params:")
    print(grid_clf.best_params_)

    return grid_clf.best_estimator_


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> List[float]:
    """
    Computes model metrics (precision, recall, and fbeta) for RandomForest.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns:
        List[float]: List of model metrics [precision, recall, fbeta]
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return [precision, recall, fbeta]


def inference(model: RandomForestClassifier, x_test: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : Trained sklearn RandomForestClassifier
    x-test : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(x_test)

    return preds


def compute_sliced_metrics(preprocessor: PreProcessor,
                           model: RandomForestClassifier):
    """
    Function to compute model metrics on data slices.

    Function loops over data categories and category values, computing model
    metrics with each category-value pair held fixed.

    Args:
        preprocessor (PreProcessor): Preprocessing class for training and
            testing data.
        model (RandomForestClassifier): Trained RandomForestClassifier

    Returns:
        None: Writes a CSV output file of sliced model metrics. 
    """

    metrics_df = pd.DataFrame(columns=["feature", "value",
                                       "precision", "recall", "fbeta"])

    for feature in preprocessor.categorical_features:
        for value in preprocessor.x_test[feature].unique():
            subset_x, subset_y = preprocessor.process_data(feature=feature,
                                                           value=value)
            preds = inference(model, subset_x)
            precision, recall, fbeta = compute_model_metrics(subset_y, preds)
            metrics_df.loc[len(metrics_df.index)] = [feature, value,
                                                     precision, recall, fbeta]

    metrics_df.to_csv("sliced_metrics.csv")

    return None
