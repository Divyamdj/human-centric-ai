import os
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import entropy


class ActiveLearner:
    def __init__(self):
        csv_path = os.path.join(settings.BASE_DIR, "media", "IMDB_Dataset.csv")
        df = pd.read_csv(
            csv_path, encoding="utf-8-sig", on_bad_lines="skip", engine="python"
        )
        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.X_texts = df["review"].tolist()
        self.y_true = df["label"].values
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X = self.vectorizer.fit_transform(self.X_texts)

        self.X_test = self.X[:1000]
        self.y_test = self.y_true[:1000]
        self.pool_idx = list(range(1000, 5000))
        self.labeled_idx = []

        self.model = LogisticRegression()

    def query(self, strategy="entropy", mode="simulated"):
        if not self.labeled_idx:
            self.labeled_idx = list(
                np.random.choice(self.pool_idx, size=20, replace=False)
            )
            for idx in self.labeled_idx:
                self.pool_idx.remove(idx)

        self.model.fit(self.X[self.labeled_idx], self.y_true[self.labeled_idx])
        probs = self.model.predict_proba(self.X[self.pool_idx])

        if strategy == "entropy":
            scores = entropy(probs.T)
        elif strategy == "margin":
            part = np.partition(-probs, 1, axis=1)
            scores = -(part[:, 0] + part[:, 1])
        else:
            scores = 1 - np.max(probs, axis=1)

        next_pool_idx = np.argmax(scores)
        real_idx = self.pool_idx[next_pool_idx]

        if mode == "simulated":
            self.labeled_idx.append(real_idx)
            self.pool_idx.remove(real_idx)
            return None

        if mode == "manual":
            return self.X_texts[real_idx], real_idx

    def label_sample(self, idx, label):
        self.y_true[idx] = int(label)
        if idx not in self.labeled_idx:
            self.labeled_idx.append(idx)
        if idx in self.pool_idx:
            self.pool_idx.remove(idx)

    def evaluate(self):
        if not self.labeled_idx:
            return 0.0
        preds = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, preds) * 100

    @property
    def n_labeled(self):
        return len(self.labeled_idx)
