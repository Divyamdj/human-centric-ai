from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os
import pickle
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from .al_logic import ActiveLearner


# Initialize learner once (simplified: you may use session or DB to persist)
learner = ActiveLearner()


# Simple index view


def index():
    return HttpResponse("Welcome to Supervised Learning!")


def train_text_classifier():
    csv_path = os.path.join(settings.BASE_DIR, "media", "IMDB_Dataset.csv")
    df = pd.read_csv(
        csv_path, encoding="utf-8-sig", on_bad_lines="skip", engine="python"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    y_train = y_train.map({"positive": 1, "negative": 0})
    y_test = y_test.map({"positive": 1, "negative": 0})

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path = os.path.join(settings.BASE_DIR, "media", "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return accuracy


# Function to load pre-trained model and evaluate


def load_and_evaluate_model():
    csv_path = os.path.join(settings.BASE_DIR, "media", "IMDB_Dataset.csv")
    df = pd.read_csv(
        csv_path, encoding="utf-8-sig", on_bad_lines="skip", engine="python"
    )

    _, X_test, _, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    y_test = y_test.map({"positive": 1, "negative": 0})

    model_path = os.path.join(settings.BASE_DIR, "media", "trained_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


def train_model_view(request):
    accuracy = None
    load_accuracy = None

    if request.method == "POST":
        if "train_model" in request.POST:
            accuracy = train_text_classifier()
        elif "load_model" in request.POST:
            load_accuracy = load_and_evaluate_model()

    return render(
        request,
        "text_classification.html",
        {"accuracy": accuracy, "load_accuracy": load_accuracy},
    )


def query_instance_view(request):
    strategy = request.POST.get("strategy", "entropy")
    mode = request.POST.get("mode", "simulated")

    if mode == "simulated":
        learner.query(strategy=strategy, mode="simulated")
        return render(
            request,
            "al_status.html",
            {"queried": learner.n_labeled, "accuracy": learner.evaluate()},
        )

    elif mode == "manual":
        sample, idx = learner.query(strategy=strategy, mode="manual")
        return render(request, "manual_label.html", {"sample_text": sample, "idx": idx})


def label_sample_view(request):
    idx = int(request.POST.get("idx"))
    label = request.POST.get("label")
    learner.label_sample(idx, label)

    return render(
        request,
        "al_status.html",
        {"queried": learner.n_labeled, "accuracy": learner.evaluate()},
    )
