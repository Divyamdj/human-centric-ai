import matplotlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse

from .forms import RegressionForm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    f1_score,
    r2_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .forms import ClassificationForm

import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.offline as opy
import plotly.graph_objects as go

from math import ceil

from statsmodels.nonparametric.smoothers_lowess import lowess


def index():
    return HttpResponse("Welcome to Supervised Learning!")


def upload_csv(request):
    result = None
    error = None
    file_url = None
    image_urls = []

    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["file"]

            # Save CSV to fixed filename
            fs = FileSystemStorage()
            filename = "uploaded_data.csv"
            if fs.exists(filename):
                fs.delete(filename)
            filename = fs.save(filename, uploaded_file)
            request.session["uploaded_file"] = filename
            file_url = fs.url(filename)

            try:
                file_path = fs.path(filename)
                df = pd.read_csv(file_path)

                corr_matrix = df.corr(numeric_only=True)

                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    title="Correlation Heatmap",
                )
                heatmap_html = pio.to_html(fig, full_html=False)

                numeric_cols = df.select_dtypes(include="number").columns

                potential_targets = df.select_dtypes(include="object").columns
                hue_col = None
                for col in potential_targets:
                    if df[col].nunique() <= 10:
                        hue_col = col
                        break

                n_cols = len(numeric_cols)
                n_rows = ceil(n_cols / 2)
                fig2 = make_subplots(rows=n_rows, cols=2, subplot_titles=numeric_cols)

                for i, col in enumerate(numeric_cols):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    hist = px.histogram(df, x=col, nbins=30)
                    for trace in hist.data:
                        fig2.add_trace(trace, row=row, col=col_pos)

                fig2.update_layout(
                    height=300 * n_rows,
                    width=900,
                    title_text="Histograms of Numeric Columns",
                    showlegend=False,
                    bargap=0.1,
                )

                pairplot_html = pio.to_html(fig2, full_html=False)

                result = identify_model_type(df)

                return render(
                    request,
                    "plots.html",
                    {
                        "result": result,
                        "error": error,
                        "heatmap_html": heatmap_html,
                        "pairplot_html": pairplot_html,
                    },
                )

            except Exception as e:
                error = f"Error processing file: {str(e)}"

    else:
        form = CSVUploadForm()
        return render(
            request,
            "csv.html",
            {},
        )


def train_regression_model(request):
    mse = None
    error_message = None
    r2 = None
    regression_plot_div = None
    selected_metric = None
    selected_metric_value = None

    if request.method == "POST":
        form = RegressionForm(request.POST, request.FILES)
        graph_type = request.POST.get("graph_type", "scatter")
        if form.is_valid():
            model_type = form.cleaned_data["model_type"]
            alpha = form.cleaned_data.get("alpha")
            max_depth = form.cleaned_data.get("max_depth")
            test_size = form.cleaned_data["test_size"]

            try:
                df = pd.read_csv("media/uploaded_data.csv")
                df = df.dropna()

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                X = pd.get_dummies(X)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size
                )

                if model_type == "linear":
                    model = LinearRegression()
                elif model_type == "ridge":
                    param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
                    model = GridSearchCV(
                        Ridge(alpha=alpha if alpha is not None else 1.0),
                        param_grid,
                        scoring="neg_mean_squared_error",
                        cv=5,
                    )
                elif model_type == "decision_tree":
                    model = DecisionTreeRegressor(
                        max_depth=max_depth if max_depth is not None else None
                    )
                else:
                    raise ValueError("Invalid model selected")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                metric_type = request.POST.get("metric_type", "mse")
                selected_metric = metric_type
                if metric_type == "mse":
                    selected_metric_value = mse
                elif metric_type == "r2":
                    selected_metric_value = r2
                else:
                    selected_metric_value = None

                if graph_type == "scatter":
                    # Scatter with LOWESS
                    sorted_idx = np.argsort(y_test)
                    y_test_sorted = np.array(y_test)[sorted_idx]
                    y_pred_sorted = np.array(y_pred)[sorted_idx]
                    smoothed = lowess(y_pred_sorted, y_test_sorted, frac=0.3)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=y_test_sorted,
                            y=y_pred_sorted,
                            mode="markers",
                            name="Actual vs Predicted",
                            marker=dict(color="blue", opacity=0.6),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=smoothed[:, 0],
                            y=smoothed[:, 1],
                            mode="lines",
                            name="LOWESS Fit",
                            line=dict(color="red", width=3),
                        )
                    )
                    fig.update_layout(
                        title="Actual vs Predicted Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values",
                        height=500,
                        width=700,
                    )
                elif graph_type == "residuals":
                    # Residuals plot
                    residuals = y_test - y_pred
                    fig = px.scatter(
                        x=y_pred,
                        y=residuals,
                        labels={"x": "Predicted Values", "y": "Residuals"},
                        title="Residuals vs Predicted",
                    )
                    fig.add_shape(
                        type="line",
                        x0=min(y_pred),
                        y0=0,
                        x1=max(y_pred),
                        y1=0,
                        line=dict(color="red", dash="dash"),
                    )
                elif graph_type == "hist_residuals":
                    # Histogram of residuals
                    residuals = y_test - y_pred
                    fig = px.histogram(
                        x=residuals,
                        nbins=30,
                        labels={"x": "Residuals"},
                        title="Histogram of Residuals",
                    )
                else:
                    fig = go.Figure()

                regression_plot_div = pio.to_html(fig, full_html=False)

                return render(
                    request,
                    "regression_results.html",
                    {
                        "mse": mse,
                        "r2": r2,
                        "error_message": error_message,
                        "regression_plot_div": regression_plot_div,
                        "selected_metric": selected_metric,
                        "selected_metric_value": selected_metric_value,
                    },
                )

            except Exception as e:
                error_message = str(e)
    else:
        form = RegressionForm()
        return render(
            request,
            "train_regression.html",
            {
                "form": form,
            },
        )


def train_classification_model(request):
    f1 = None
    accuracy = None
    error_message = None
    plot_div = None
    selected_metric = None
    selected_metric_value = None

    if request.method == "POST":
        form = ClassificationForm(request.POST, request.FILES)
        graph_type = request.POST.get("graph_type", "scatter")
        if form.is_valid():
            model_type = form.cleaned_data["model_type"]
            test_size = form.cleaned_data["test_size"]
            n_estimators = form.cleaned_data.get("n_estimators")
            c_param = form.cleaned_data.get("c_param")
            kernel = form.cleaned_data.get("kernel")
            learning_rate = form.cleaned_data.get("learning_rate")
            max_depth_xgb = form.cleaned_data.get("max_depth_xgb")
            max_depth_rf = form.cleaned_data.get("max_depth_rf")

            try:
                df = pd.read_csv("media/uploaded_data.csv")
                df = df.dropna()

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                X = pd.get_dummies(X)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, stratify=y
                )

                if model_type == "random_forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators if n_estimators is not None else 100,
                        max_depth=max_depth_rf if max_depth_rf is not None else None,
                    )
                elif model_type == "svm":
                    model = SVC(
                        C=c_param if c_param is not None else 1.0,
                        kernel=kernel if kernel is not None else "rbf",
                    )
                elif model_type == "xg_boost":
                    model = XGBClassifier(
                        learning_rate=(
                            learning_rate if learning_rate is not None else 0.3
                        ),
                        max_depth=max_depth_xgb if max_depth_xgb is not None else 6,
                        use_label_encoder=False,
                        eval_metric="mlogloss",
                    )
                else:
                    raise ValueError("Invalid model selected")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1 = f1_score(y_test, y_pred, average="weighted")
                accuracy = accuracy_score(y_test, y_pred)

                metric_type = request.POST.get("metric_type", "accuracy")
                selected_metric = metric_type
                if metric_type == "accuracy":
                    selected_metric_value = accuracy
                elif metric_type == "f1":
                    selected_metric_value = f1
                else:
                    selected_metric_value = None

                if graph_type == "scatter":
                    # Scatter plot: color by true class, symbol by predicted class
                    if X_test.shape[1] >= 2:
                        x_axis = X_test[:, 0]
                        y_axis = X_test[:, 1]
                        feature_names = X.columns[:2]
                    else:
                        x_axis = X_test[:, 0]
                        y_axis = X_test[:, 0]
                        feature_names = [X.columns[0], X.columns[0]]

                    scatter_df = pd.DataFrame(
                        {
                            feature_names[0]: x_axis,
                            feature_names[1]: y_axis,
                            "True Class": y_test.astype(str).values,
                            "Predicted Class": y_pred.astype(str),
                        }
                    )

                    fig = px.scatter(
                        scatter_df,
                        x=feature_names[0],
                        y=feature_names[1],
                        color="True Class",
                        symbol="Predicted Class",
                        title="True vs Predicted Classes",
                        labels={"color": "True Class", "symbol": "Predicted Class"},
                        opacity=0.8,
                    )
                elif graph_type == "hist":
                    # Histogram of predicted classes
                    fig = px.histogram(
                        x=y_pred.astype(str),
                        labels={"x": "Predicted Class", "y": "Count"},
                        title="Histogram of Predicted Classes",
                    )
                elif graph_type == "cmatrix":
                    # Confusion Matrix Heatmap
                    cm = confusion_matrix(y_test, y_pred)
                    labels = sorted(list(set(y_test) | set(y_pred)))
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                    fig = px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted Label", y="True Label", color="Count"),
                        title="Confusion Matrix Heatmap",
                    )
                    fig.update_layout(
                        width=900,  # Increase width
                        height=900,  # Increase height
                        margin=dict(
                            l=120, r=120, t=120, b=120
                        ),  # More margin for labels
                    )
                    fig.update_xaxes(
                        tickangle=45, tickfont=dict(size=14), automargin=True
                    )
                    fig.update_yaxes(tickfont=dict(size=14), automargin=True)
                else:
                    fig = go.Figure()

                plot_div = opy.plot(fig, auto_open=False, output_type="div")

            except Exception as e:
                error_message = str(e)

            return render(
                request,
                "classification_results.html",
                {
                    "f1": f1,
                    "accuracy": accuracy,
                    "error_message": error_message,
                    "plot_div": plot_div,
                    "selected_metric": selected_metric,
                    "selected_metric_value": selected_metric_value,
                },
            )
    else:
        form = ClassificationForm()
        return render(
            request,
            "train_classification.html",
            {
                "form": form,
            },
        )


def identify_model_type(df):
    if df.iloc[:, -1].dtype == "object" or df.iloc[:, -1].nunique() < 10:
        return "Classification"
    elif pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
        return "Regression"
