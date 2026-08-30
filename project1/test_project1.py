from . import views
from unittest.mock import MagicMock
import pandas as pd


def test_index_smoke():
    response = views.index()
    assert response is not None

def test_upload_csv_smoke():
    request = MagicMock()
    response = views.upload_csv(request)
    assert response is not None

def test_train_regression_model_smoke():
    request = MagicMock()
    request.method = "GET"
    response = views.train_regression_model(request)
    assert response is not None

def test_train_classification_model_smoke():
    request = MagicMock()
    request.method = "GET"
    response = views.train_classification_model(request)
    assert response is not None

def test_identify_model_type_smoke():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = views.identify_model_type(df)
    assert result in ["Classification", "Regression"]
