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
    print(r)
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 42"}


# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200


# def test_model_predicts_over_50k(salary_over_50k: Person):
#     """
#     Function to test that expected columns are present in the dataframe.

#     Args:
#         preprocessor (PreProcessor): Trained data Preprocessor.
#     """

#     url = "https://reqres.in/api/users"
#     response = requests.get(url)
#     assert response.status_code == 200
