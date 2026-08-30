import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

import plotly.express as px
import plotly.io as pio

from django.shortcuts import render, redirect
from palmerpenguins import load_penguins

from sklearn.tree import plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import numpy as np
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.views.decorators.http import require_http_methods


from gosdt import (
    GOSDTClassifier,  # main classifier
    ThresholdGuessBinarizer,  # alternative binarizer
)

PENGUINS_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"


def index(request):
    if request.method == "POST":
        return redirect("project3:decision_tree")

    try:
        penguins = load_penguins()
    except ImportError:
        penguins = pd.read_csv(PENGUINS_URL)

    fig = px.scatter(
        penguins.dropna(),
        x="bill_length_mm",
        y="flipper_length_mm",
        color="species",
        symbol="sex",
        title="Palmer Penguins: Bill Length vs Flipper Length",
    )
    penguins_plot_div = pio.to_html(fig, full_html=False)
    return render(request, "penguins.html", {"penguins_plot_div": penguins_plot_div})


def decision_tree_view(request):
    tree_plot = None
    accuracy = None
    n_leaves = None
    tree_text = None

    try:
        penguins = load_penguins()
    except ImportError:
        penguins = pd.read_csv(PENGUINS_URL)

    penguins = penguins.dropna()
    X = penguins[
        [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island",
            "sex",
            "year",
        ]
    ]
    y = penguins["species"]

    X = pd.get_dummies(X, columns=["island", "sex", "year"])
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    n_leaves = clf.get_n_leaves()
    tree_text = export_text(clf, feature_names=list(X.columns))

    plt.figure(figsize=(14.5, 9), dpi=100)
    plot_tree(
        clf,
        feature_names=list(X.columns),
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=False,
        proportion=False,
        class_names=le.classes_,
        label="all",
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    tree_plot = f"data:image/png;base64,{image_base64}"

    return render(
        request,
        "decision_tree.html",
        {
            "tree_plot": tree_plot,
            "accuracy": accuracy,
            "n_leaves": n_leaves,
            "tree_text": tree_text,
        },
    )


def sparse_tree_view(request):
    lambda_str = request.GET.get("lambda", None)

    # Default values for template
    context = {
        "accuracy": None,
        "lambda_value": 0.01,
        "rules": None,
        "error_message": None,
    }

    if lambda_str is None:
        # No lambda param given, just show form without training
        return render(request, "sparse_tree.html", context)

    try:
        lambda_value = float(lambda_str)
        lambda_value = max(0.001, min(lambda_value, 1.0))  # clamp lambda
        context["lambda_value"] = lambda_value
    except ValueError:
        context["error_message"] = "Invalid lambda value."
        return render(request, "sparse_tree.html", context)

    try:
        penguins = load_penguins()
    except ImportError:
        penguins = pd.read_csv(PENGUINS_URL)
    df = penguins.dropna()
    X = df[
        [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island",
            "sex",
            "year",
        ]
    ]
    y = df["species"]
    X = pd.get_dummies(X, columns=["island", "sex", "year"])
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    # Threshold binarization for GOSDT
    enc = ThresholdGuessBinarizer()
    enc.set_output(transform="pandas")
    X_train_bin = enc.fit_transform(X_train, y_train)
    X_test_bin = enc.transform(X_test)

    # Train GOSDT model with error handling
    try:
        clf = GOSDTClassifier(
            regularization=lambda_value, depth_budget=6, time_limit=10, verbose=False
        )
        clf.fit(X_train_bin, y_train)
        y_pred = clf.predict(X_test_bin)
        accuracy = accuracy_score(y_test, y_pred)
        context["accuracy"] = accuracy

        # Train surrogate decision tree on original features but using GOSDT predictions on training binarized data
        surrogate = DecisionTreeClassifier(max_depth=4, random_state=42)
        surrogate.fit(X_train, clf.predict(X_train_bin))
        rules = export_text(surrogate, feature_names=list(X.columns))
        context["rules"] = rules
    except Exception as e:
        context["error_message"] = f"GOSDT failed for lambda={lambda_value}: {str(e)}"

    return render(request, "sparse_tree.html", context)


def logistic_regression_view(request):
    try:
        penguins = load_penguins()
    except ImportError:
        penguins = pd.read_csv(PENGUINS_URL)

    df = penguins.dropna()
    all_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex",
        "year",
    ]
    selected_features = all_features

    if request.method == "POST":
        selected_features = request.POST.getlist("selected_features")
        if not selected_features:
            selected_features = all_features

    X = df[selected_features]
    y = df["species"]

    categorical = [f for f in ["island", "sex", "year"] if f in selected_features]
    X = pd.get_dummies(X, columns=categorical)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    used_features = (clf.coef_ != 0).any(axis=0)
    used_feature_names = [name for name, used in zip(X.columns, used_features) if used]
    used_original_features = set()
    for name in used_feature_names:
        for orig in selected_features:
            if name.startswith(orig):
                used_original_features.add(orig)
                break
    n_used_original_features = len(used_original_features)

    user_n_used_features = None
    feedback = None
    if request.method == "POST":
        user_n_used_features = request.POST.get("n_used_features")
        if user_n_used_features is not None:
            try:
                user_n_used_features = int(user_n_used_features)
                if user_n_used_features == n_used_original_features:
                    feedback = f"Correct! The model used {n_used_original_features} original features."
                else:
                    feedback = f"Incorrect. The model used {n_used_original_features} original features."
            except ValueError:
                feedback = "Please enter a valid number."

    return render(
        request,
        "logistic_regression.html",
        {
            "accuracy": accuracy,
            "n_used_features": n_used_original_features,
            "used_feature_names": used_original_features,
            "user_n_used_features": user_n_used_features,
            "feedback": feedback,
            "available_features": all_features,
            "selected_features": selected_features,
        },
    )


def mad(series):
    """Median absolute deviation (MAD) for a pandas Series (non-zero floor)."""
    med = series.median()
    mad_val = (series - med).abs().median()
    return max(mad_val, 1e-6)


def mad_weighted_l1_distance(x, x_cf, numeric_cols, categorical_cols, mad_values):
    """
    Weighted L1: for numeric features use |diff| / MAD.
    For categorical, contribution is 0 if equal else 1 (could also use 1/M where M=1).
    """
    num_diff = np.abs(
        x[numeric_cols].astype(float) - x_cf[numeric_cols].astype(float)
    ).fillna(0)
    num_term = (
        (num_diff / mad_values).sum(axis=1)
        if isinstance(num_diff, pd.DataFrame)
        else (num_diff / mad_values).sum()
    )
    cat_term = (x[categorical_cols] != x_cf[categorical_cols]).astype(int).sum(axis=1)
    return np.array(num_term) + np.array(cat_term)


def sample_local_points(
    x_row,
    df,
    numeric_cols,
    categorical_cols,
    N=200,
    numeric_scale=0.5,
    cat_change_prob=0.15,
    rng=None,
):
    """
    Create N local samples around x_row in the original (non-dummified) feature space.
    - numeric features: sample from normal centered at value with sigma = numeric_scale * MAD(feature)
    - categorical features: with probability cat_change_prob pick another category (sampled from observed distribution)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    samples = []
    mad_values = df[numeric_cols].apply(mad)
    category_levels = {c: df[c].dropna().unique().tolist() for c in categorical_cols}

    for _ in range(N):
        new = x_row.copy()
        for c in numeric_cols:
            val = x_row[c]
            sigma = numeric_scale * mad_values[c]
            if np.isnan(val):
                val = df[c].median()
            new[c] = float(rng.normal(loc=float(val), scale=float(sigma)))
        for c in categorical_cols:
            if rng.rand() < cat_change_prob:
                choices = category_levels[c]
                if len(choices) > 1:
                    other_choices = [v for v in choices if v != x_row[c]]
                    if other_choices:
                        new[c] = rng.choice(other_choices)
        samples.append(new)
    samples_df = pd.DataFrame(samples, columns=list(x_row.index))
    return samples_df, mad_values


def prepare_for_model(df_raw, model_feature_columns):
    """One-hot encode and align to model_feature_columns (reindex missing columns with zeros)."""
    X = pd.get_dummies(
        df_raw,
        columns=[
            c
            for c in df_raw.columns
            if df_raw[c].dtype == "object"
            or str(df_raw[c].dtype).startswith("category")
        ],
    )
    X = X.reindex(columns=model_feature_columns, fill_value=0)
    return X


def counterfactual_view(request):
    try:
        penguins = load_penguins()
    except ImportError:
        penguins = pd.read_csv(PENGUINS_URL)
    df = penguins.dropna().reset_index(drop=True)

    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex",
        "year",
    ]
    numeric_cols = [
        c
        for c in feature_cols
        if df[c].dtype != "object" and not str(df[c].dtype).startswith("category")
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X_full = pd.get_dummies(df[feature_cols], columns=categorical_cols)
    y_full = df["species"]
    le = LabelEncoder()
    y_full_enc = le.fit_transform(y_full)

    clf = DecisionTreeClassifier(random_state=0, max_depth=6)
    clf.fit(X_full, y_full_enc)
    model_feature_columns = list(X_full.columns)

    indices = list(df.index)
    classes = list(le.classes_)
    results = None

    form_values = {
        "index": 0,
        "target_label": classes[0],
        "N": 200,
        "k": 5,
        "numeric_scale": 0.5,
        "cat_change_prob": 0.15,
    }

    if request.method == "POST":
        idx = int(request.POST.get("example_index", 0))
        idx = max(0, min(idx, len(df) - 1))
        target_label = request.POST.get("target_label", classes[0])
        N = int(request.POST.get("N", 200))
        k = int(request.POST.get("k", 5))
        numeric_scale = float(request.POST.get("numeric_scale", 0.5))
        cat_change_prob = float(request.POST.get("cat_change_prob", 0.15))

        form_values.update(
            {
                "index": idx,
                "target_label": target_label,
                "N": N,
                "k": k,
                "numeric_scale": numeric_scale,
                "cat_change_prob": cat_change_prob,
            }
        )

        x_row = df.loc[idx, feature_cols].copy()

        samples_df, mad_values = sample_local_points(
            x_row,
            df[feature_cols],
            numeric_cols,
            categorical_cols,
            N=N,
            numeric_scale=numeric_scale,
            cat_change_prob=cat_change_prob,
            rng=np.random.RandomState(0),
        )

        X_samples = prepare_for_model(samples_df, model_feature_columns)
        y_preds = clf.predict(X_samples)
        desired_code = int(np.where(le.classes_ == target_label)[0][0])

        mask_desired = y_preds == desired_code
        if mask_desired.sum() > 0:
            candidate_cf = samples_df[mask_desired].reset_index(drop=True)
            x_repeat = pd.DataFrame([x_row.values], columns=x_row.index).astype(object)
            mad_vec = mad_values[numeric_cols]
            distances = mad_weighted_l1_distance(
                x_repeat.loc[x_repeat.index.repeat(len(candidate_cf))].reset_index(
                    drop=True
                ),
                candidate_cf.reset_index(drop=True),
                numeric_cols,
                categorical_cols,
                mad_vec,
            )
            candidate_cf["distance"] = distances
            results = (
                candidate_cf.sort_values("distance").reset_index(drop=True).head(k)
            )

    return render(
        request,
        "counterfactuals.html",
        {
            "indices": indices,
            "classes": classes,
            "form_values": form_values,
            "results": results,
            "example_preview": df.loc[form_values["index"], feature_cols].to_dict(),
        },
    )
