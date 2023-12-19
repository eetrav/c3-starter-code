# Script to train machine learning model.
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add the necessary imports for the starter code.

# Add code to load in the data.
census_df = pd.read_csv('../data/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(census_df, test_size=0.20)

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

# Process the training data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save the model
model = train_model(X_train, y_train)
joblib.dump(model, "../tests/test_model.pkl")

# Perform inference on the trained model
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(precision, recall, fbeta)
