"""
Pytest module to test input inference API.

Author: Emily Travinsky
Date: 01/2024
"""

from fastapi.testclient import TestClient
from starter.main import app
from starter.main import Person
import pytest

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to our salary prediction model!"}


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200


def test_model_predicts_over_50k(salary_over_50k: Person):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        preprocessor (PreProcessor): Trained data Preprocessor.
    """

    r = client.post("/prediction/", salary_over_50k)
    print(r)
    # assert response.status_code == 200
