"""Binary direction (Up/Down) classification metrics.

Rows where either the predicted or actual label is missing/"Unknown" are
dropped before scoring -- a model that abstains is not penalized the same
way as one that actively predicts the wrong direction, but it also isn't
scored as if it were correct. The number of dropped rows is reported so
this doesn't silently inflate apparent accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

UP, DOWN = "Up", "Down"


@dataclass
class ClassificationResult:
    n_obs: int
    n_dropped: int
    accuracy: float
    balanced_accuracy: float
    up_recall: float
    down_recall: float
    down_precision: float
    false_alarm_rate: float
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "n_dropped": self.n_dropped,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "up_recall": self.up_recall,
            "down_recall": self.down_recall,
            "down_precision": self.down_precision,
            "false_alarm_rate": self.false_alarm_rate,
            "confusion_matrix": self.confusion_matrix,
        }


def _align_and_clean(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    df = pd.concat({"true": y_true, "pred": y_pred}, axis=1, join="inner")
    n_total = len(df)
    df = df[df["true"].isin([UP, DOWN]) & df["pred"].isin([UP, DOWN])]
    n_dropped = n_total - len(df)
    return df["true"], df["pred"], n_dropped


def evaluate_binary_direction(y_true: pd.Series, y_pred: pd.Series) -> ClassificationResult:
    true_clean, pred_clean, n_dropped = _align_and_clean(y_true, y_pred)

    if len(true_clean) == 0:
        empty_cm = {UP: {UP: 0, DOWN: 0}, DOWN: {UP: 0, DOWN: 0}}
        nan = np.nan
        return ClassificationResult(0, n_dropped, nan, nan, nan, nan, nan, nan, empty_cm)

    labels = [DOWN, UP]
    cm = confusion_matrix(true_clean, pred_clean, labels=labels)
    cm_dict = {
        DOWN: {DOWN: int(cm[0, 0]), UP: int(cm[0, 1])},
        UP: {DOWN: int(cm[1, 0]), UP: int(cm[1, 1])},
    }

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / cm.sum() if cm.sum() > 0 else np.nan
    balanced_accuracy = balanced_accuracy_score(true_clean, pred_clean)
    up_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    down_recall = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    down_precision = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return ClassificationResult(
        n_obs=len(true_clean),
        n_dropped=n_dropped,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        up_recall=up_recall,
        down_recall=down_recall,
        down_precision=down_precision,
        false_alarm_rate=false_alarm_rate,
        confusion_matrix=cm_dict,
    )


def naive_baseline(y_true: pd.Series, mode: str) -> ClassificationResult:
    """mode: 'always_up' or 'always_down'."""
    if mode not in ("always_up", "always_down"):
        raise ValueError("mode must be 'always_up' or 'always_down'")
    label = UP if mode == "always_up" else DOWN
    y_pred = pd.Series(label, index=y_true.index)
    return evaluate_binary_direction(y_true, y_pred)
