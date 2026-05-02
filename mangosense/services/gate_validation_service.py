from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mangosense.ml_constants import (
    GATE_CONFIDENCE_THRESHOLD,
    GATE_CONFIDENCE_THRESHOLD_FRUIT,
    GATE_CONFIDENCE_THRESHOLD_LEAF,
    GATE_FRUIT_CLASS_NAMES,
    GATE_LEAF_CLASS_NAMES,
    GATE_VALID_INDEX_FRUIT,
    GATE_VALID_INDEX_LEAF,
)
from mangosense.services.model_loader_service import load_model


@dataclass
class GateResult:
    passed: bool
    gate_prediction_label: Optional[str]
    gate_predicted_confidence: Optional[float]
    mango_confidence: Optional[float]
    threshold: float


def run_gate_validation(tf, img_array, gate_model_path: str, detection_type: str) -> GateResult:
    if not os.path.exists(gate_model_path):
        print(f"Gate model not found at {gate_model_path} — skipping validation")
        return GateResult(
            passed=True,
            gate_prediction_label=None,
            gate_predicted_confidence=None,
            mango_confidence=None,
            threshold=GATE_CONFIDENCE_THRESHOLD,
        )

    try:
        gate_model = load_model(gate_model_path, tf)
        gate_pred = gate_model.predict(img_array)
        gate_pred = np.array(gate_pred).flatten()

        if detection_type == 'fruit':
            valid_idx = GATE_VALID_INDEX_FRUIT
            gate_cls = GATE_FRUIT_CLASS_NAMES
            threshold = GATE_CONFIDENCE_THRESHOLD_FRUIT
        else:
            valid_idx = GATE_VALID_INDEX_LEAF
            gate_cls = GATE_LEAF_CLASS_NAMES
            threshold = GATE_CONFIDENCE_THRESHOLD_LEAF

        gate_predicted_idx = int(np.argmax(gate_pred))
        gate_prediction_label = gate_cls[gate_predicted_idx]
        gate_predicted_confidence = float(gate_pred[gate_predicted_idx]) * 100
        mango_confidence = float(gate_pred[valid_idx]) * 100

        gate_passed = (gate_predicted_idx == valid_idx) and (mango_confidence >= threshold)

        print(f"=== GATE VALIDATION DEBUG ===")
        print(f"Detection type: {detection_type}")
        print(f"Gate predictions: {dict(zip(gate_cls, [f'{p*100:.2f}%' for p in gate_pred]))}")
        print(f"Predicted class: {gate_prediction_label} ({gate_predicted_confidence:.2f}%)")
        print(f"Mango confidence: {mango_confidence:.2f}%")
        print(f"Threshold: {threshold}%")
        print(f"GATE PASSED: {gate_passed}")
        print(f"=============================")

        del gate_model
        gc.collect()

        return GateResult(
            passed=gate_passed,
            gate_prediction_label=gate_prediction_label,
            gate_predicted_confidence=gate_predicted_confidence,
            mango_confidence=mango_confidence,
            threshold=threshold,
        )

    except Exception as gate_err:
        print(f"Gate model error: {gate_err} — skipping validation")
        return GateResult(
            passed=True,
            gate_prediction_label=None,
            gate_predicted_confidence=None,
            mango_confidence=None,
            threshold=GATE_CONFIDENCE_THRESHOLD,
        )
