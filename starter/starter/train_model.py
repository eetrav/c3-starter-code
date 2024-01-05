"""
Script to train machine learning model.

Author: Emily Travinsky
Date: 12/2023
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from ml.data import PreProcessor
from ml.model import train_model, inference
from ml.model import compute_model_metrics, compute_sliced_metrics

# Add the necessary imports for the starter code.

# Add code to load in the data.
census_df = pd.read_csv('../data/census_cleaned.csv')

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

# Optional: use K-fold cross validation instead of train-test split.
preprocessor = PreProcessor(census_df,
                            categorical_features=cat_features,
                            label="salary",
                            training=True,
                            )
preprocessor.train_test_split(
    test_size=0.20, stratify_by=census_df["sex"])

# Process the training data with the process_data function.
X_train, y_train = preprocessor.process_data()

# Process the test data with the process_data function.
preprocessor.training = False
X_test, y_test = preprocessor.process_data()

# Train and save the model
model = train_model(X_train, y_train)
# joblib.dump(model, "../tests/test_model.pkl")

# Perform inference on the trained model
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Precision: {0:2f}".format(precision))
print("Recall: {0:2f}".format(recall))
print("fbeta: {0:2f}".format(fbeta))

# Compute metrics by data slicing and store in sliced_metrics.csv
compute_sliced_metrics(preprocessor, model)
