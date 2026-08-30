from . import views
from unittest.mock import MagicMock


def test_index_smoke():
    response = views.index()
    assert response is not None

def test_train_text_classifier_smoke():
    result = views.train_text_classifier()
    assert result is not None

def test_load_and_evaluate_model_smoke():
    result = views.load_and_evaluate_model()
    assert result is not None

def test_train_model_view_smoke():
    request = MagicMock()
    request.method = "GET"
    response = views.train_model_view(request)
    assert response is not None

def test_query_instance_view_smoke():
    request = MagicMock()
    request.POST.get = lambda key, default=None: default
    response = views.query_instance_view(request)
    assert response is not None

def test_label_sample_view_smoke():
    request = MagicMock()
    request.POST.get = lambda key, default=None: "0" if key == "idx" else "1"
    response = views.label_sample_view(request)
    assert response is not None
