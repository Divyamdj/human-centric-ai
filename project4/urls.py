from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/candidates/", views.get_movie_candidates_view, name="get_candidates"),
    path("api/predict-impact/", views.predict_rating_impact, name="predict_impact"),
    path(
        "api/final-recommendations/",
        views.get_final_recommendations,
        name="final_recommendations",
    ),
    path("download-matrix-pdf/", views.download_pdf, name="download_matrix_pdf"),
]
