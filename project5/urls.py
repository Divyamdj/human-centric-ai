# urls.py
from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path("", views.dashboard, name="index"),
    path("sample/", views.sample_policy, name="sample_policy"),
    path("train/", views.train, name="train"),
    path("reset/", views.reset_env, name="reset_env"),
    path("run/", views.run_episode, name="run_episode"),
    path("compare/", views.compare_trajectories, name="compare"),
    path("reward-train/", views.train_reward_model, name="reward_train"),
    path("finetune/", views.fine_tune_policy, name="finetune")
]
