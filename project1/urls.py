from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path("", views.upload_csv, name="index"),
    path("upload/", views.upload_csv, name="upload_csv"),
    path("regression/", views.train_regression_model, name="regression"),
    path("classification/", views.train_classification_model, name="classification"),
]
