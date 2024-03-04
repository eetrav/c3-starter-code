import requests

api_site = 'https://c3-starter-code.onrender.com/'

features = {
    "age": 52,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "45781",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capitol-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}

response = requests.post(api_site, data=features)
