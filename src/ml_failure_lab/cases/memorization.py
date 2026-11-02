"""Evaluating on rows used for fitting mistakes memorization for generalization."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class MemorizationRun:
    fit_rows: frozenset[int]
    evaluated_rows: frozenset[int]
    accuracy: float


def experiment(approach: str, seed: int = 42) -> MemorizationRun:
    if approach not in {"bad", "fixed"}:
        raise ValueError(f"Unknown approach: {approach}")

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(100, 20))
    targets = rng.integers(0, 2, size=100)

    row_ids = np.arange(len(targets))
    train_ids, test_ids = train_test_split(
        row_ids, test_size=0.3, random_state=seed, stratify=targets
    )

    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(features[train_ids], targets[train_ids])

    eval_ids = train_ids if approach == "bad" else test_ids
    predictions = tree.predict(features[eval_ids])
    accuracy = float(accuracy_score(targets[eval_ids], predictions))

    return MemorizationRun(
        fit_rows=frozenset(int(i) for i in train_ids),
        evaluated_rows=frozenset(int(i) for i in eval_ids),
        accuracy=accuracy,
    )


def require_disjoint_evaluation(result: MemorizationRun) -> None:
    overlap = result.fit_rows & result.evaluated_rows
    if overlap:
        raise ValueError(
            f"Evaluation must not evaluate on fitted rows ({len(overlap)} overlapping records)."
        )


def run() -> dict:
    measurements = []
    for seed in (42, 43, 44):
        for approach in ("bad", "fixed"):
            res = experiment(approach, seed=seed)
            measurements.append(
                {
                    "approach": approach,
                    "seed": seed,
                    "accuracy": res.accuracy,
                    "overlapping_rows": len(res.fit_rows & res.evaluated_rows),
                }
            )

    return {
        "case": "memorization",
        "question": "Was model accuracy measured on held-out records not used for fitting?",
        "measurements": measurements,
        "requirement": (
            "Evaluation records must be disjoint from records used during fitting."
        ),
        "interpretation": (
            "An unpruned decision tree fits synthetic random binary noise to 100% accuracy on "
            "training records. Evaluating on those same records measures memorization, not "
            "unseen generalization. A disjoint held-out test split reflects true predictive capability."
        ),
        "limitation": (
            "Synthetic random features and labels demonstrate extreme memorization. Exact held-out "
            "scores vary by library version and seed, but the requirement targets record isolation."
        ),
    }


def verify(approach: str) -> None:
    require_disjoint_evaluation(experiment(approach))
