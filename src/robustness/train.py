"""Train replacement reference models on original train/val only, by authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.eval.metrics import LABELS, macro_f1
from src.robustness.inference import sha256
from src.robustness.reference import ReferenceModel


def train_reference(data_dir: Path, models_dir: Path, seed: int = 1729) -> dict:
    """Select hyperparameters using validation only. Never read test/challenge."""
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"
    train, validation = pd.read_csv(train_path), pd.read_csv(val_path)
    for frame in (train, validation):
        if (
            frame.empty
            or set(frame["label"]) != set(LABELS)
            or frame["text"].isna().any()
        ):
            raise ValueError(
                "Reference training needs nonempty texts and all four labels"
            )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=20000,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(validation["text"])
    vocabulary = {
        str(word): int(index) for word, index in vectorizer.vocabulary_.items()
    }
    idf = vectorizer.idf_.tolist()
    vectorizer_hash = hashlib.sha256(
        json.dumps({"vocabulary": vocabulary, "idf": idf}, sort_keys=True).encode()
    ).hexdigest()
    models_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for family in ("logistic_regression", "naive_bayes"):
        name = "reference_" + family
        path = models_dir / name / f"seed{seed}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen weights: {path}")
        candidates = []
        for parameter in (0.5, 1.0, 2.0):
            estimator = (
                LogisticRegression(
                    C=parameter,
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=2000,
                )
                if family == "logistic_regression"
                else MultinomialNB(alpha=parameter)
            )
            estimator.fit(x_train, train["label"])
            score = macro_f1(validation["label"], estimator.predict(x_val))
            candidates.append((score, parameter, estimator))
        score, parameter, estimator = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        order = [list(estimator.classes_).index(label) for label in LABELS]
        weights = (
            estimator.coef_
            if family == "logistic_regression"
            else estimator.feature_log_prob_
        )[order]
        bias = (
            estimator.intercept_
            if family == "logistic_regression"
            else estimator.class_log_prior_
        )[order]
        payload = {
            "schema": 1,
            "name": name,
            "seed": seed,
            "labels": list(LABELS),
            "vocabulary": vocabulary,
            "idf": idf,
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "vectorizer_sha256": vectorizer_hash,
            "validation_macro_f1": score,
            "training": {
                "train_sha256": sha256(train_path),
                "validation_sha256": sha256(val_path),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "selected_parameter": parameter,
                "candidate_validation_scores": [
                    {"parameter": value, "macro_f1": value_score}
                    for value_score, value, _ in candidates
                ],
                "test_used_for_selection": False,
                "challenge_used_for_training": False,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
        loaded = ReferenceModel(path)
        np.testing.assert_allclose(
            loaded.probabilities(validation["text"].tolist()),
            estimator.predict_proba(x_val)[:, order],
            atol=1e-10,
        )
        result[name] = {
            "path": str(path),
            "validation_macro_f1": score,
            "checkpoint_sha256": loaded.identity["checkpoint_sha256"],
        }
    return result
