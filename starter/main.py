from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

import joblib


# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int


# Save items from POST method in the memory
items = {}

# Initialize FastAPI instance
app = FastAPI()
model = joblib.load("./model/test_model.pkl")


def hyphenize(field: str):
    return field.replace("_", "-")


class Person(BaseModel):
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

    # https://github.com/pydantic/pydantic/issues/2266
    class Config:
        alias_generator = hyphenize


@app.get("/")
async def model_greeting():
    return {"greeting": "Welcome to our salary prediction model!"}

# https://www.amplemarket.com/blog/serving-machine-learning-models-with-fastapi
# This allows sending of data (our Person) via POST to the API.


@app.post("/prediction/")
async def predict_salary(person: Person):
    prediction = model.predict(person)
    return prediction

# # A GET that in this case just returns the item_id we pass,
# # but a future iteration may link the item_id here to the one we defined in our TaggedItem.
# @app.get("/items/{item_id}")
# async def get_items(item_id: int, count: int = 1):
#     try:
#         item = items[item_id]
#     except:
#         return "Item not found."

#     return {"fetch": f"Fetched: {item.name} with qty of {count}"}
