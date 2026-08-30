from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("decision-tree/", views.decision_tree_view, name="decision_tree"),
    path("sparse/", views.sparse_tree_view, name="sparse_tree"),
    path("logistic-regression/", views.logistic_regression_view, name="logistic_regression",),
    path("counterfactuals/", views.counterfactual_view, name="counterfactuals"),
]
