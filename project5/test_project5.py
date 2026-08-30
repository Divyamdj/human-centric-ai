import sys
sys.modules["django.views.decorators.csrf"].csrf_protect = lambda x: x

import pytest
from django.http import HttpRequest
from django.test import RequestFactory
from unittest.mock import patch, MagicMock

import project5.views as views

@pytest.mark.django_db
def test_dashboard(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    monkeypatch.setattr(views, "GRID", [[0]])
    monkeypatch.setattr(views, "render_board", lambda grid: [["▫️"]])
    monkeypatch.setattr("os.path.exists", lambda path: True)
    request = HttpRequest()
    response = views.dashboard(request)

@pytest.mark.django_db
def test_sample_policy(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    mock_trainer = MagicMock()
    mock_trainer.obs_from_grid.return_value = 0
    mock_trainer.select_action.return_value = 0
    monkeypatch.setattr(views, "TRAINER", mock_trainer)
    monkeypatch.setattr(views, "GRID", [[0]])
    monkeypatch.setattr("project5.mouse.move", lambda action, grid: ([[0]], 1))
    monkeypatch.setattr(views, "render_board", lambda grid: [["▫️"]])
    request = HttpRequest()
    request.method = "POST"
    response = views.sample_policy(request)

@pytest.mark.django_db
def test_train(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {"history": [{"epoch": 1, "loss": 0.1, "avg_return": 1.0}]}
    monkeypatch.setattr(views, "TRAINER", mock_trainer)
    request = RequestFactory().post("/train", data={"epochs": "1", "episodes_per_epoch": "1"})
    response = views.train(request)

@pytest.mark.django_db
def test_reset_env(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    monkeypatch.setattr("project5.mouse.initialize_grid_with_cheese_types", lambda: ([[0]], None, None, None))
    request = RequestFactory().post("/reset_env")
    response = views.reset_env(request)

@pytest.mark.django_db
def test_run_episode(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    mock_trainer = MagicMock()
    mock_trainer.generate_episode.return_value = (None, None, [{"reward": 1, "grid": [["▫️"]], "action": "UP", "explanation": "Moved"}], None)
    monkeypatch.setattr(views, "TRAINER", mock_trainer)
    request = RequestFactory().post("/run_episode")
    response = views.run_episode(request)

@pytest.mark.django_db
def test_compare_trajectories_get(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    mock_trainer = MagicMock()
    mock_trainer.generate_episode.return_value = (None, None, [{"reward": 1, "grid": [["▫️"]], "action": "UP", "explanation": "Moved"}], None)
    monkeypatch.setattr(views, "TRAINER", mock_trainer)
    request = RequestFactory().get("/compare_trajectories")
    request.session = {}
    response = views.compare_trajectories(request)

@pytest.mark.django_db
def test_compare_trajectories_post(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    request = RequestFactory().post("/compare_trajectories", data={"choice": "traj1"})
    request.session = {"traj1": [{"reward": 1, "grid": [["▫️"]], "action": "UP", "explanation": "Moved"}],
                       "traj2": [{"reward": 0, "grid": [["▫️"]], "action": "DOWN", "explanation": "Moved"}]}
    monkeypatch.setattr("os.path.exists", lambda path: False)
    response = views.compare_trajectories(request)

@pytest.mark.django_db
def test_train_reward_model_no_feedback(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    request = RequestFactory().get("/train_reward_model")
    response = views.train_reward_model(request)

@pytest.mark.django_db
def test_train_reward_model_empty_feedback(monkeypatch, tmp_path):
    # Create an empty feedback file
    feedback_file = tmp_path / "feedback_log.json"
    feedback_file.write_text("[]")
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("builtins.open", lambda f, mode="r": feedback_file.open(mode))
    monkeypatch.setattr("project5.views.feedback_file", str(feedback_file))
    request = RequestFactory().get("/train_reward_model")
    response = views.train_reward_model(request)

@pytest.mark.django_db
def test_fine_tune_policy_get(monkeypatch):
    monkeypatch.setattr(views, "initialize_global_state", lambda: None)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("torch.load", lambda path: None)
    monkeypatch.setattr("project5.views.policy_base_path", "dummy")
    monkeypatch.setattr("project5.views.reward_model_path", "dummy")
    request = RequestFactory().get("/fine_tune_policy")
    response = views.fine_tune_policy(request)