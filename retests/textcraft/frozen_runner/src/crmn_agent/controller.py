"""Explicit, inspectable reliability signals and a distilled softmax prior."""

import json
import math
from pathlib import Path

import numpy as np

from .schemas import Anchor, Audit, Reconstruction


def cycle_error(anchors: list[Anchor], reconstruction: Reconstruction) -> float:
    """Only grounded, CURRENT anchors are scored. Unobserved facts are never false."""
    if not anchors:
        return 0.0
    decoded = {f.key.strip().casefold(): f.value.strip().casefold() for f in reconstruction.facts}
    return sum(decoded.get(a.key.strip().casefold()) != a.value.strip().casefold()
               for a in anchors) / len(anchors)


def prediction_error(audit: Audit, observation: str, predictions: list[str]) -> tuple[float, int]:
    """An unknown outcome, missing verdict or ungrounded quotation gives no teaching signal."""
    checked = {}
    for verdict in audit.verdicts:
        if verdict.prediction not in predictions or verdict.prediction in checked:
            continue
        if verdict.status == "unknown":
            continue
        if not verdict.evidence.strip() or verdict.evidence not in observation:
            continue
        checked[verdict.prediction] = verdict.status == "contradicted"
    return (sum(checked.values()) / len(checked), len(checked)) if checked else (0.0, 0)


def responsibilities(errors: list[float], priors: list[float], values: list[float],
                     precision: float = 3.0, utility: float = 0.5) -> list[float]:
    if not errors or not (len(errors) == len(priors) == len(values)):
        raise ValueError("Equal, nonempty signal vectors required")
    if any(not math.isfinite(v) for v in errors + priors + values):
        raise ValueError("Signals must be finite")
    if any(p < 0 for p in priors) or sum(priors) <= 0:
        raise ValueError("Priors must be nonnegative with positive total")
    logits = np.log(np.maximum(priors, 1e-9)) - precision * np.asarray(errors) + utility * np.asarray(values)
    weights = np.exp(logits - np.max(logits))
    return (weights / weights.sum()).tolist()


def entropy(weights: list[float]) -> float:
    return -sum(p * math.log(p) for p in weights if p > 0)


def features(observation: str, step: int, recent_actions: list[str]) -> np.ndarray:
    return np.asarray([
        1.0, min(len(observation) / 2000, 2), min(step / 50, 2),
        float("carry" in observation.lower() or "inventory" in observation.lower()),
        float("nothing happens" in observation.lower() or "invalid" in observation.lower()),
        float(len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1),
    ])


class Prior:
    def __init__(self):
        self.weights = np.zeros((2, 6))
        self.training_rows = 0

    def predict(self, x: np.ndarray) -> list[float]:
        logits = self.weights @ x
        probs = np.exp(logits - logits.max())
        return (probs / probs.sum()).tolist()

    def fit(self, rows: list[dict], epochs: int = 100, learning_rate: float = 0.05):
        if not rows or any(r.get("split") != "train" for r in rows):
            raise ValueError("The prior accepts nonempty TRAINING rows only; no validation or test fitting")
        x = np.asarray([r["features"] for r in rows])
        y = np.asarray([r["responsibilities"] for r in rows])
        if x.shape[1:] != (6,) or y.shape != (len(rows), 2):
            raise ValueError("Invalid training feature/target shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid training values")
        if not np.allclose(y.sum(axis=1), 1):
            raise ValueError("Responsibility targets must sum to one")
        for _ in range(epochs):
            logits = x @ self.weights.T
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            predicted = exp / exp.sum(axis=1, keepdims=True)
            self.weights -= learning_rate * ((predicted - y).T @ x / len(rows))
        self.training_rows = len(rows)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": self.weights.tolist(), "training_rows": self.training_rows}))

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        obj = cls()
        obj.weights = np.asarray(data["weights"], dtype=float)
        if obj.weights.shape != (2, 6) or not np.isfinite(obj.weights).all():
            raise ValueError("Invalid prior checkpoint")
        obj.training_rows = data["training_rows"]
        return obj

