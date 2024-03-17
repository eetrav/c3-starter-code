import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, ConfigDict

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

# Initialize FastAPI instance
app = FastAPI()

# Load pre-computed encoder and pre-trained model
encoder = joblib.load("./starter/tests/encoder.pkl")
model = joblib.load("./starter/tests/test_model.pkl")


def hyphenize(field: str) -> str:
    """
    Function to generate an alias for data input fields, replacing _ with -

    Args:
        field (str): field to hyphenize

    Returns:
        str: field with underscores replaced by hyphens
    """

    return field.replace("_", "-")


def convert_pred_to_val(prediction: int) -> str:

    pred_keys = {0: "<= $50K",
                 1: "> $50K"}

    return pred_keys[prediction]


class Person(BaseModel):
    """
    Class to describe a Person used as input to salary prediction model.

    Args:
        BaseModel (BaseModel): Inheritance from Pydantic BaseModel
    """

    model_config = ConfigDict(alias_generator=hyphenize)

    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 2174
    capitol_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"


@app.get("/")
async def model_greeting() -> dict:
    """
    Root get function for API greeting.

    Returns:
        dict: JSON dict output containing API greeting
    """
    return {"greeting": "Welcome to our salary prediction model!"}

# https://www.amplemarket.com/blog/serving-machine-learning-models-with-fastapi
# This allows sending of data (our Person) via POST to the API.


# def get_salary(person: Person) -> str:
#     print(person)
#     sample_df = pd.DataFrame(person.dict(by_alias=True), index=[0])
#     print(sample_df)
#     x_categorical = sample_df[cat_features].values
#     x_continuous = sample_df.drop(*[cat_features], axis=1)
#     x_categorical = encoder.transform(x_categorical)
#     sample = np.concatenate([x_continuous, x_categorical], axis=1)
#     prediction = model.predict(sample)
#     salary = convert_pred_to_val(prediction[0])
#     return salary


@app.post("/prediction/")
async def predict_salary(person: Person):
    """
    API POST function to run model prediction on Person descriptor.

    Args:
        person (Person): Features for person to predict salary

    Returns:
        dict: Model salary prediction
    """
    sample_df = pd.DataFrame(person.dict(by_alias=True), index=[0])
    x_categorical = sample_df[cat_features].values
    x_continuous = sample_df.drop(*[cat_features], axis=1)
    x_categorical = encoder.transform(x_categorical)
    sample = np.concatenate([x_continuous, x_categorical], axis=1)
    prediction = model.predict(sample)
    salary = convert_pred_to_val(prediction[0])
    return {"prediction": salary}
