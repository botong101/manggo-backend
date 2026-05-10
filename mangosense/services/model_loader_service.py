from __future__ import annotations


def get_tensorflow_runtime():
    try:
        import tensorflow as tf
        return tf, None
    except Exception as exc:
        return None, str(exc)


def load_model(model_path: str, tf):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mobilenet_preprocess
    _custom_objects = {'preprocess_input': _mobilenet_preprocess}

    # Attempt 1: supply custom_objects (resolves Lambda(preprocess_input))
    try:
        return tf.keras.models.load_model(model_path, custom_objects=_custom_objects)
    except (TypeError, ValueError):
        pass

    # Attempt 2: also patch Dense.from_config to drop quantization_config
    Dense = tf.keras.layers.Dense
    _orig_fn = Dense.from_config.__func__

    @classmethod
    def _compat_from_config(cls, config):
        config.pop('quantization_config', None)
        return _orig_fn(cls, config)

    Dense.from_config = _compat_from_config
    try:
        return tf.keras.models.load_model(model_path, custom_objects=_custom_objects)
    finally:
        Dense.from_config = classmethod(_orig_fn)
