"""
MangoSenseNet-CoAttn (Hybrid CNN) training worker.
Ported from train-custom-cnn-symptom.py for the Django backend.

Uses images downloaded from the database and a pre-extracted symptom CSV
to train the dual-input co-attention model.
"""
import datetime
import json
import os
import shutil
import tempfile

import numpy as np

from .config import RetrainConfig
from .dataset import build_temp_dataset, collect_verified_images
from .state import _set

META_COLUMNS = {
    'dataset', 'split', 'class', 'filename',
    'filepath', 'relative_path', 'color_dominant_hue_bin',
}

IMG_SIZE = (224, 224)


def _build_model(input_shape, num_symptom_features, num_classes):
    import tensorflow as tf
    _REG       = tf.keras.regularizers.l2(1e-4)
    TOKEN_DIM  = 16
    IMG_TOKENS = 16
    SYM_TOKENS = 4
    NUM_HEADS  = 4
    KEY_DIM    = TOKEN_DIM // NUM_HEADS

    def _conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=_REG)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        return x

    img_input = tf.keras.Input(shape=input_shape, name='image')
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ], name='augmentation')(img_input)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = _conv_block(x, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)
    x = tf.keras.layers.GlobalAveragePooling2D(name='image_gap')(x)
    x = tf.keras.layers.Dense(IMG_TOKENS * TOKEN_DIM, activation='relu', kernel_regularizer=_REG)(x)
    x = tf.keras.layers.Dropout(0.5, name='img_dropout')(x)
    img_tokens = tf.keras.layers.Reshape((IMG_TOKENS, TOKEN_DIM), name='img_tokens')(x)

    sym_input = tf.keras.Input(shape=(num_symptom_features,), name='symptoms')
    s = tf.keras.layers.BatchNormalization()(sym_input)
    s = tf.keras.layers.Dense(SYM_TOKENS * TOKEN_DIM, activation='relu', kernel_regularizer=_REG)(s)
    s = tf.keras.layers.Dropout(0.3, name='sym_dropout')(s)
    sym_tokens = tf.keras.layers.Reshape((SYM_TOKENS, TOKEN_DIM), name='sym_tokens')(s)

    img_attended = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=KEY_DIM, name='img_attends_sym',
    )(query=img_tokens, key=sym_tokens, value=sym_tokens)
    sym_attended = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=KEY_DIM, name='sym_attends_img',
    )(query=sym_tokens, key=img_tokens, value=img_tokens)

    img_out = tf.keras.layers.LayerNormalization(name='img_layernorm')(img_tokens + img_attended)
    sym_out = tf.keras.layers.LayerNormalization(name='sym_layernorm')(sym_tokens + sym_attended)

    img_flat = tf.keras.layers.Dropout(0.3)(tf.keras.layers.Flatten(name='img_flat')(img_out))
    sym_flat = tf.keras.layers.Dropout(0.3)(tf.keras.layers.Flatten(name='sym_flat')(sym_out))
    fused    = tf.keras.layers.Concatenate(name='fusion')([img_flat, sym_flat])
    out      = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=_REG)(fused)
    out      = tf.keras.layers.Dropout(0.4)(out)
    outputs  = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(out)

    return tf.keras.Model(
        inputs={'image': img_input, 'symptoms': sym_input},
        outputs=outputs,
        name='MangoSenseNet_CoAttn',
    )


def _compute_class_means(df, feature_cols, class_names):
    means = df.groupby('class')[feature_cols].mean()
    return np.array([
        means.loc[c].values if c in means.index else np.zeros(len(feature_cols))
        for c in class_names
    ], dtype=np.float32)


def _attach_class_symptoms(class_feat_tensor, training=False, dropout_rate=0.5):
    import tensorflow as tf

    def _fn(image, label):
        class_idx = tf.argmax(label, axis=-1)
        symptoms  = tf.gather(class_feat_tensor, class_idx)
        if training:
            keep     = tf.cast(tf.random.uniform([tf.shape(symptoms)[0], 1]) > dropout_rate, tf.float32)
            symptoms = symptoms * keep
        return {'image': image, 'symptoms': symptoms}, label
    return _fn


def run_hybrid_retraining(model_type: str, output_path: str, config: RetrainConfig) -> None:
    tmp_dir = None
    try:
        import joblib
        import pandas as pd
        import tensorflow as tf
        from django.conf import settings as django_settings
        from sklearn.preprocessing import StandardScaler

        AUTOTUNE = tf.data.AUTOTUNE

        # ── 1. Download verified + training_ready images ──────────────────────
        _set(phase='downloading', progress=5,
             message='Downloading verified + training_ready images…')
        class_map = collect_verified_images(model_type, config.min_images_per_class)
        if len(class_map) < 2:
            raise ValueError(
                f'Not enough eligible classes. Need 2+ classes with '
                f'{config.min_images_per_class}+ images. '
                f'Found: {list(class_map.keys()) or "none"}'
            )
        total_images = sum(len(v) for v in class_map.values())
        _set(progress=10,
             message=f'Found {total_images} images across {len(class_map)} classes: {list(class_map.keys())}')

        # ── 2. Load symptom CSV ───────────────────────────────────────────────
        _set(phase='preparing', progress=12, message='Loading symptom feature CSV…')
        csv_path = os.path.join(
            django_settings.MEDIA_ROOT, 'symptom_data', f'{model_type}_symptom_features.csv'
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f'Symptom CSV not found: {csv_path}. '
                'Run "Extract Features" (Step 1) before retraining.'
            )
        df = pd.read_csv(csv_path)
        feature_cols = [
            c for c in df.columns
            if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = df.dropna(subset=feature_cols)
        if df.empty:
            raise ValueError('Symptom CSV has no usable rows after dropping NaN values.')
        _set(progress=14,
             message=f'Symptom CSV loaded — {len(df)} rows, {len(feature_cols)} features.')

        # ── 3. Build temp train/val dataset ───────────────────────────────────
        _set(progress=16, message='Preparing dataset directory…')
        tmp_dir = tempfile.mkdtemp(prefix='mangosense_hybrid_retrain_')
        train_dir, val_dir, dataset_info = build_temp_dataset(class_map, tmp_dir, config.val_split)
        _set(progress=20, dataset_info=dataset_info, message='Dataset directory ready.')

        # ── 4. Create tf.data datasets ────────────────────────────────────────
        _set(phase='training', progress=23, message='Loading TF image datasets…')
        train_ds_raw = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=IMG_SIZE, batch_size=config.batch_size,
            label_mode='categorical', shuffle=True, seed=42,
        )
        val_ds_raw = tf.keras.utils.image_dataset_from_directory(
            val_dir, image_size=IMG_SIZE, batch_size=config.batch_size,
            label_mode='categorical', shuffle=False,
        )
        class_names = train_ds_raw.class_names
        num_classes = len(class_names)
        _set(progress=28, message=f'Classes ({num_classes}): {class_names}')

        # ── 5. Compute class-mean symptom prototypes ──────────────────────────
        _set(progress=30, message='Computing class symptom prototypes…')
        class_means_raw    = _compute_class_means(df, feature_cols, class_names)
        scaler             = StandardScaler()
        class_means_scaled = scaler.fit_transform(class_means_raw).astype(np.float32)
        class_feat_tensor  = tf.constant(class_means_scaled)
        num_features       = len(feature_cols)
        _set(progress=33, message=f'{num_features} symptom features. Modality dropout: {config.modality_dropout}')

        # ── 6. Build model ────────────────────────────────────────────────────
        _set(progress=35, message='Building MangoSenseNet-CoAttn model…')
        model = _build_model((*IMG_SIZE, 3), num_features, num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        _set(progress=38, message=f'Model compiled — {model.count_params():,} parameters.')

        # ── 7. Attach symptom prototypes to datasets ──────────────────────────
        train_attach  = _attach_class_symptoms(class_feat_tensor, training=True,
                                               dropout_rate=config.modality_dropout)
        oracle_attach = _attach_class_symptoms(class_feat_tensor, training=False)

        train_ds = (
            train_ds_raw.cache().shuffle(1000, seed=42)
            .map(train_attach,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        )
        val_ds = (
            val_ds_raw.cache()
            .map(oracle_attach, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        )

        # ── 8. Train ──────────────────────────────────────────────────────────
        total_epochs = config.epochs

        class _ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                _set(
                    progress=40 + int((epoch + 1) / total_epochs * 45),
                    message=(
                        f'Epoch {epoch + 1}/{total_epochs} — '
                        f'train_acc: {logs.get("accuracy", 0.0):.4f}, '
                        f'val_acc: {logs.get("val_accuracy", 0.0):.4f}'
                    ),
                )

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=total_epochs,
            callbacks=[
                _ProgressCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=config.early_stopping_patience,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=config.lr_reduce_factor,
                    patience=config.lr_reduce_patience, verbose=0,
                ),
            ],
        )

        # ── 9. Evaluate ───────────────────────────────────────────────────────
        _set(phase='evaluating', progress=87, message='Evaluating on validation set…')
        _, val_acc = model.evaluate(val_ds, verbose=0)
        final_acc  = round(float(val_acc) * 100, 2)
        _set(progress=93, accuracy=final_acc,
             message=f'Evaluation done — val_accuracy: {val_acc:.4f} ({final_acc}%)')

        # ── 10. Save model + artifacts ────────────────────────────────────────
        _set(phase='saving', progress=95, message='Saving model and artifacts…')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)

        artifacts_dir = os.path.dirname(output_path)
        stem          = os.path.splitext(os.path.basename(output_path))[0]
        joblib.dump(scaler, os.path.join(artifacts_dir, f'{stem}_scaler.joblib'))
        with open(os.path.join(artifacts_dir, f'{stem}_feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)
        class_proto = {c: class_means_scaled[i].tolist() for i, c in enumerate(class_names)}
        with open(os.path.join(artifacts_dir, f'{stem}_prototypes.json'), 'w') as f:
            json.dump(class_proto, f)

        output_filename = os.path.basename(output_path)
        _set(
            phase='done', progress=100, is_running=False,
            finished_at=datetime.datetime.now().isoformat(),
            output_filename=output_filename,
            message=f'Hybrid CNN training complete. Model saved as "{output_filename}".',
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set(
            phase='error', is_running=False,
            finished_at=datetime.datetime.now().isoformat(),
            error=str(exc),
            message=f'Hybrid CNN training failed: {exc}',
        )
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
