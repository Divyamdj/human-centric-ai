from . import views
from unittest.mock import MagicMock
import pandas as pd


def test_index_smoke():
    request = MagicMock()
    request.method = "GET"
    response = views.index(request)
    assert response is not None

def test_decision_tree_view_smoke():
    request = MagicMock()
    response = views.decision_tree_view(request)
    assert response is not None

def test_sparse_tree_view_smoke():
    request = MagicMock()
    request.GET.get = lambda key, default=None: None
    response = views.sparse_tree_view(request)
    assert response is not None

def test_logistic_regression_view_smoke():
    request = MagicMock()
    response = views.logistic_regression_view(request)
    assert response is not None

def test_mad_smoke():
    series = pd.Series([1, 2, 3, 4, 5])
    result = views.mad(series)
    assert isinstance(result, float)

def test_mad_weighted_l1_distance_smoke():
    x = pd.DataFrame({"a": [1.0], "b": [2.0]})
    x_cf = pd.DataFrame({"a": [1.5], "b": [2.5]})
    mad_values = pd.Series([1.0, 1.0], index=["a", "b"])
    result = views.mad_weighted_l1_distance(x, x_cf, ["a", "b"], [], mad_values)
    assert result is not None

def test_prepare_for_model_smoke():
    df_raw = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
    model_feature_columns = ["cat_a", "cat_b", "num"]
    result = views.prepare_for_model(df_raw, model_feature_columns)
    assert result is not None

def test_counterfactual_view_smoke():
    request = MagicMock()
    response = views.counterfactual_view(request)
    assert response is not None
