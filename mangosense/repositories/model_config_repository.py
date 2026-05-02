from __future__ import annotations

import os

from django.conf import settings

from mangosense.ml_constants import (
    _DEFAULT_FRUIT_MODEL,
    _DEFAULT_GATE_FRUIT_MODEL,
    _DEFAULT_GATE_LEAF_MODEL,
    _DEFAULT_LEAF_MODEL,
)


def get_active_model_path(detection_type: str, is_gate: bool = False) -> str:
    if is_gate:
        config_key = f'gate_{detection_type}'
        default = _DEFAULT_GATE_LEAF_MODEL if detection_type == 'leaf' else _DEFAULT_GATE_FRUIT_MODEL
    else:
        config_key = detection_type
        default = _DEFAULT_LEAF_MODEL if detection_type == 'leaf' else _DEFAULT_FRUIT_MODEL

    try:
        from mangosense.models import ModelConfig
        config = ModelConfig.objects.get(detection_type=config_key)
        filename = config.model_filename
    except Exception:
        filename = default

    return os.path.join(settings.BASE_DIR, 'models', filename)
