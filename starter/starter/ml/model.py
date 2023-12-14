from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model and returns it.

    Constructs a model pipeline to: 
    1) Performs feature pre-processing to normalize and zero-center 
    non-categorical variables
    2) Train a Logistic Regression model with cross-fold validation. 

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """

    # From scikit-learn LogisticRegressionCV documentation:
    # ‘newton-cholesky’ is a good choice for n_samples >> n_features,
    # especially with one-hot encoded categorical features with rare categories.
    pipe = make_pipeline(
        StandardScaler(), LogisticRegressionCV(solver="newton-cholesky"))

    pipe.fit(X_train, y_train)

    return pipe


def compute_model_metrics(y, preds):
    """
    Validates the trained ML model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, x_test):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model.base.LinearRegressionCV
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(x_test)

    return preds
