import pytest
from django.http import HttpRequest, JsonResponse
from django.test import RequestFactory
from unittest.mock import patch, MagicMock
import numpy as np
import json

import project4.views as views

@pytest.mark.django_db
def test_index_view():
    request = HttpRequest()
    response = views.index(request)
    assert response.status_code == 200

@pytest.mark.django_db
def test_download_pdf(monkeypatch):
    request = HttpRequest()
    # Patch open to avoid file IO
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    monkeypatch.setattr("os.path.join", lambda *a: "dummy_path")
    monkeypatch.setattr("django.conf.settings.BASE_DIR", "dummy_base")
    response = views.download_pdf(request)
    assert hasattr(response, "streaming_content")

@pytest.mark.django_db
def test_get_movie_candidates_view(monkeypatch):
    request = HttpRequest()
    # Patch recommender and its method
    monkeypatch.setattr(views, "recommender", MagicMock())
    views.recommender.get_movie_candidates.return_value = [
        {"movieId": 1, "title": "Test Movie", "genres": "Action", "avg_rating": 4.0, "n_ratings": 100}
    ]
    response = views.get_movie_candidates_view(request)
    assert isinstance(response, JsonResponse)
    assert response.status_code == 200

@pytest.mark.django_db
def test_predict_rating_impact(monkeypatch):
    factory = RequestFactory()
    data = {
        "user_ratings": {"1": 5.0},
        "movie_id": 2,
        "rating": 4.0
    }
    request = factory.post("/predict_rating_impact", data=json.dumps(data), content_type="application/json")
    monkeypatch.setattr(views, "recommender", MagicMock())
    views.recommender.predict_impact.return_value = [
        {"title": "Movie X", "predicted_rating": 4.5, "genres": "Comedy", "confidence": 0.8, "num_similar_users": 10, "ranking_score": 4.5}
    ]
    response = views.predict_rating_impact(request)
    assert isinstance(response, JsonResponse)
    assert response.status_code == 200

@pytest.mark.django_db
def test_get_final_recommendations(monkeypatch):
    factory = RequestFactory()
    data = {
        "user_ratings": {"1": 5.0, "2": 3.0}
    }
    request = factory.post("/get_final_recommendations", data=json.dumps(data), content_type="application/json")
    monkeypatch.setattr(views, "recommender", MagicMock())
    views.recommender.predict_impact.return_value = [
        {"title": "Movie Y", "predicted_rating": 4.0, "genres": "Drama", "confidence": 0.7, "num_similar_users": 5, "ranking_score": 4.0}
    ]
    views.recommender._get_fallback_recommendations.return_value = [
        {"title": "Fallback Movie", "predicted_rating": 3.5, "genres": "Action", "confidence": 0.3, "num_similar_users": 0}
    ]
    response = views.get_final_recommendations(request)
    assert isinstance(response, JsonResponse)
    assert response.status_code == 200

@pytest.mark.django_db
def test_movie_recommender_predict_rating():
    # Properly mock the DataFrame chain to return a numpy array for .values
    fake_values = np.array([[5.0]])
    fake_df = MagicMock()
    fake_df.index = [1]
    fake_df.columns = [1]
    fake_df.values = fake_values
    # fillna returns fake_df itself
    fake_df.fillna.return_value = fake_df

    # pivot_table returns fake_df
    fake_ratings = MagicMock()
    fake_ratings.pivot_table.return_value = fake_df

    with patch.object(views, "load_movielens_data", return_value=(fake_ratings, MagicMock())):
        recommender = views.MovieRecommender()
        recommender.user_id_map = {1: 0}
        recommender.movie_id_map = {1: 0}
        recommender.U = recommender.V = [[1.0]]
        assert recommender.predict_rating(1, 1) >= 0.5