from __future__ import annotations


def get_tensorflow_runtime():
    try:
        import tensorflow as tf
        return tf, None
    except Exception as exc:
        return None, str(exc)


def load_model(model_path: str, tf):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as first_load_error:
        if 'quantization_config' not in str(first_load_error):
            raise
        _orig_dense_init = tf.keras.layers.Dense.__init__
        def _compat_dense_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            _orig_dense_init(self, *args, **kwargs)
        tf.keras.layers.Dense.__init__ = _compat_dense_init
        try:
            return tf.keras.models.load_model(model_path)
        finally:
            tf.keras.layers.Dense.__init__ = _orig_dense_init
