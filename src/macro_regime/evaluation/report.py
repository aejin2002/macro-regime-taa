"""Assemble per-model evaluation results (vs. naive baselines) into a
single report structure that the CLI and Streamlit app both consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from macro_regime.evaluation.classification import (
    ClassificationResult,
    evaluate_binary_direction,
    naive_baseline,
)
from macro_regime.evaluation.lead_lag import average_lead_time_months


@dataclass
class ModelEvaluation:
    model_name: str
    target_name: str
    horizon_months: int
    result: ClassificationResult
    baseline_always_up: ClassificationResult
    baseline_always_down: ClassificationResult
    average_lead_months: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_name": self.target_name,
            "horizon_months": self.horizon_months,
            "result": self.result.to_dict(),
            "baseline_always_up": self.baseline_always_up.to_dict(),
            "baseline_always_down": self.baseline_always_down.to_dict(),
            "average_lead_months": self.average_lead_months,
        }


@dataclass
class EvaluationReport:
    evaluations: list[ModelEvaluation] = field(default_factory=list)

    def add(self, evaluation: ModelEvaluation) -> None:
        self.evaluations.append(evaluation)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for ev in self.evaluations:
            rows.append(
                {
                    "model": ev.model_name,
                    "target": ev.target_name,
                    "horizon_months": ev.horizon_months,
                    "n_obs": ev.result.n_obs,
                    "accuracy": ev.result.accuracy,
                    "balanced_accuracy": ev.result.balanced_accuracy,
                    "up_recall": ev.result.up_recall,
                    "down_recall": ev.result.down_recall,
                    "down_precision": ev.result.down_precision,
                    "false_alarm_rate": ev.result.false_alarm_rate,
                    "baseline_always_up_accuracy": ev.baseline_always_up.accuracy,
                    "baseline_always_down_accuracy": ev.baseline_always_down.accuracy,
                    "average_lead_months": ev.average_lead_months,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"evaluations": [ev.to_dict() for ev in self.evaluations]}


def evaluate_model(
    model_name: str,
    target_name: str,
    horizon_months: int,
    predicted_label: pd.Series,
    actual_label: pd.Series,
    *,
    signal_score: pd.Series | None = None,
    target_level: pd.Series | None = None,
) -> ModelEvaluation:
    result = evaluate_binary_direction(actual_label, predicted_label)
    up_base = naive_baseline(actual_label, "always_up")
    down_base = naive_baseline(actual_label, "always_down")

    lead = None
    if signal_score is not None and target_level is not None:
        lead = average_lead_time_months(signal_score, target_level)

    return ModelEvaluation(
        model_name=model_name,
        target_name=target_name,
        horizon_months=horizon_months,
        result=result,
        baseline_always_up=up_base,
        baseline_always_down=down_base,
        average_lead_months=lead,
    )
