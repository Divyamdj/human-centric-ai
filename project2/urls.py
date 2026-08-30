from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path("", views.train_model_view, name="index"),
    path("classifier/", views.train_model_view, name="train_model"),
    path("query/", views.query_instance_view, name="query_instance"),
    path("label/", views.label_sample_view, name="label_sample"),
]
