from django import forms


class CSVUploadForm(forms.Form):
    file = forms.FileField(label="Select a CSV file")


MODEL_CHOICES_regression = [
    ("linear", "Linear Regression"),
    ("ridge", "Ridge Regression"),
    ("decision_tree", "Decision Tree Regressor"),
]

MODEL_CHOICES_classification = [
    ("random_forest", "Random Forest"),
    ("svm", "Support Vector Machines"),
    ("xg_boost", "eXtreme Gradient Boosting"),
]


class RegressionForm(forms.Form):
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES_regression, label="Regression Model"
    )
    alpha = forms.FloatField(label="Alpha (for Ridge)", required=False)
    max_depth = forms.IntegerField(
        label="Max Depth (for Decision Tree)", required=False
    )
    test_size = forms.FloatField(label="Test Size", min_value=0.1, max_value=0.9)


class ClassificationForm(forms.Form):
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES_classification,
        label="Classification Model",
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    n_estimators = forms.IntegerField(
        label="Number of Trees (Random Forest)",
        min_value=1,
        required=False,
        initial=100,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    max_depth_rf = forms.IntegerField(
        label="Max Depth (Random Forest)",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    c_param = forms.FloatField(
        label="C Parameter (SVM)",
        min_value=0.01,
        required=False,
        initial=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    kernel = forms.ChoiceField(
        label="Kernel (SVM)",
        choices=[("linear", "Linear"), ("rbf", "RBF"), ("poly", "Polynomial")],
        required=False,
        initial="rbf",
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    learning_rate = forms.FloatField(
        label="Learning Rate (XGBoost)",
        min_value=0.01,
        max_value=1.0,
        required=False,
        initial=0.1,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    max_depth_xgb = forms.IntegerField(
        label="Max Depth (XGBoost)",
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    test_size = forms.FloatField(
        label="Test Size",
        min_value=0.1,
        max_value=0.9,
        initial=0.2,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
