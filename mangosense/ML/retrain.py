import os
import shutil
import tempfile
import threading
import datetime
import random
import numpy as np
from django.conf import settings

# ── module-level job state ─────────────────────────────────────────────────────
# NOTE: this is in-process state. Works fine for single-worker deployments
# (e.g. gunicorn --workers 1). Multi-worker deployments should migrate this
# state to a database row or cache backend.

_lock = threading.Lock()
_status: dict = {
    'is_running':      False,
    'model_type':      None,    # 'leaf' | 'fruit'
    'phase':           None,    # preparing | training | evaluating | saving | done | error
    'progress':        0,       # 0–100
    'message':         '',
    'started_at':      None,
    'finished_at':     None,
    'output_filename': None,
    'accuracy':        None,
    'error':           None,
    'dataset_info':    None,    # {class_label: {total, train, val}}
}

# Minimum images per class for a class to be included in retraining
MIN_IMAGES_PER_CLASS = 5


# ── status helpers ─────────────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return dict(_status)


def _set(**kwargs):
    with _lock:
        _status.update(kwargs)


# ── dataset collection ─────────────────────────────────────────────────────────

def _collect_verified_images(model_type: str) -> dict:
    """
    Query MangoImage for verified images of the given model_type.

    Returns:
        {class_label: [absolute_file_path, ...]}
    Only classes with >= MIN_IMAGES_PER_CLASS files are included.
    """
    from ..models import MangoImage

    qs = MangoImage.objects.filter(
        is_verified=True,
        disease_type=model_type,
    ).exclude(disease_classification='').exclude(disease_classification__isnull=True)

    class_map: dict = {}
    media_root = settings.MEDIA_ROOT

    for img in qs:
        label = img.disease_classification.strip()
        if not label:
            continue
        abs_path = os.path.join(media_root, str(img.image))
        if os.path.isfile(abs_path):
            class_map.setdefault(label, []).append(abs_path)

    return {k: v for k, v in class_map.items() if len(v) >= MIN_IMAGES_PER_CLASS}


def get_dataset_preview(model_type: str) -> dict:
    """
    Return a preview of available verified images without starting training.
    Used by the dataset-info endpoint.
    """
    from ..models import MangoImage

    # all verified images for this type (regardless of min threshold)
    qs = MangoImage.objects.filter(
        is_verified=True,
        disease_type=model_type,
    ).exclude(disease_classification='').exclude(disease_classification__isnull=True)

    all_counts: dict = {}
    media_root = settings.MEDIA_ROOT

    for img in qs:
        label = img.disease_classification.strip()
        if not label:
            continue
        abs_path = os.path.join(media_root, str(img.image))
        if os.path.isfile(abs_path):
            all_counts[label] = all_counts.get(label, 0) + 1

    eligible = {k: v for k, v in all_counts.items() if v >= MIN_IMAGES_PER_CLASS}
    can_retrain = len(eligible) >= 2

    return {
        'model_type':             model_type,
        'all_classes':            all_counts,
        'eligible_classes':       eligible,
        'total_eligible_images':  sum(eligible.values()),
        'min_images_per_class':   MIN_IMAGES_PER_CLASS,
        'can_retrain':            can_retrain,
        'reason': (
            None if can_retrain
            else f'Need at least 2 classes with {MIN_IMAGES_PER_CLASS}+ verified images each.'
        ),
    }


# ── temp dataset builder ───────────────────────────────────────────────────────

def _build_temp_dataset(class_map: dict, tmp_dir: str, val_split: float = 0.2):
    """
    Copies verified images into a Keras-ready directory structure:

        tmp_dir/train/<class>/<image>
        tmp_dir/val/<class>/<image>

    Returns (train_dir, val_dir, dataset_info)
    """
    train_dir = os.path.join(tmp_dir, 'train')
    val_dir   = os.path.join(tmp_dir, 'val')
    dataset_info = {}

    for label, paths in class_map.items():
        shuffled = paths[:]
        random.shuffle(shuffled)

        n_val   = max(1, int(len(shuffled) * val_split))
        n_train = len(shuffled) - n_val

        splits = [('train', shuffled[:n_train]), ('val', shuffled[n_train:])]

        for split_name, split_paths in splits:
            dest_dir = os.path.join(tmp_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            for src in split_paths:
                base, ext = os.path.splitext(os.path.basename(src))
                dst = os.path.join(dest_dir, f"{base}_{id(src)}{ext}")
                shutil.copy2(src, dst)

        dataset_info[label] = {
            'total': len(shuffled),
            'train': n_train,
            'val':   len(shuffled) - n_train,
        }

    return train_dir, val_dir, dataset_info


# ── training ───────────────────────────────────────────────────────────────────

def _run_retraining(model_type: str, base_model_path: str, output_path: str):
    """
    Full retraining pipeline — runs inside a background thread.
    Updates module-level _status at every major step.
    """
    tmp_dir = None
    try:
        import tensorflow as tf

        # ── 1. Collect verified images ────────────────────────────────────────
        _set(phase='preparing', progress=5, message='Collecting verified images from database…')
        class_map = _collect_verified_images(model_type)

        if len(class_map) < 2:
            raise ValueError(
                f"Not enough eligible classes. "
                f"Need at least 2 classes with {MIN_IMAGES_PER_CLASS}+ verified images each. "
                f"Found: {list(class_map.keys()) or 'none'}"
            )

        total_images = sum(len(v) for v in class_map.values())
        _set(
            progress=10,
            message=f'Found {total_images} images across {len(class_map)} classes: {list(class_map.keys())}',
        )

        # ── 2. Build temp dataset directory ──────────────────────────────────
        _set(phase='preparing', progress=15, message='Preparing dataset directory…')
        tmp_dir = tempfile.mkdtemp(prefix='mangosense_retrain_')
        train_dir, val_dir, dataset_info = _build_temp_dataset(class_map, tmp_dir)
        _set(progress=20, dataset_info=dataset_info, message='Dataset directory ready.')

        # ── 3. Create tf.data datasets ────────────────────────────────────────
        _set(phase='training', progress=25, message='Loading TF datasets…')
        IMG_SIZE   = (224, 224)
        BATCH_SIZE = 16

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=True,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
        )

        num_classes = len(train_ds.class_names)
        class_names = train_ds.class_names
        _set(progress=30, message=f'Classes ({num_classes}): {class_names}')

        # ── 4. Load or build model ────────────────────────────────────────────
        _set(progress=35, message='Loading base model…')

        if os.path.exists(base_model_path):
            model = tf.keras.models.load_model(base_model_path)

            # If the number of output classes changed, rebuild the classification head
            current_output_classes = model.output_shape[-1]
            if current_output_classes != num_classes:
                _set(
                    message=(
                        f'Output class count changed '
                        f'({current_output_classes} → {num_classes}). '
                        'Rebuilding classification head…'
                    )
                )
                # Keep all layers except the last Dense (softmax) layer
                x = model.layers[-2].output
                outputs = tf.keras.layers.Dense(
                    num_classes, activation='softmax', name='retrain_predictions'
                )(x)
                model = tf.keras.Model(inputs=model.input, outputs=outputs)
        else:
            # No existing model found — build a fresh MobileNetV2 model
            _set(message='Base model not found. Building new MobileNetV2 model…')
            data_aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
            base = tf.keras.applications.MobileNetV2(
                include_top=False,
                input_shape=(224, 224, 3),
                weights='imagenet',
                pooling='avg',
            )
            base.trainable = False
            inputs  = tf.keras.Input(shape=(224, 224, 3))
            x       = data_aug(inputs)
            x       = tf.keras.layers.Rescaling(1.0 / 255)(x)
            x       = base(x, training=False)
            x       = tf.keras.layers.Dense(128, activation='relu')(x)
            x       = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model   = tf.keras.Model(inputs, outputs)

        # ── 5. Freeze most layers; unfreeze the top few for fine-tuning ───────
        total_layers  = len(model.layers)
        freeze_until  = max(0, total_layers - 20)
        for i, layer in enumerate(model.layers):
            layer.trainable = i >= freeze_until

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        _set(progress=40, message='Model compiled. Starting fine-tuning…')

        # ── 6. Training callbacks ──────────────────────────────────────────────
        EPOCHS = 10

        class _ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs: int):
                super().__init__()
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                logs     = logs or {}
                pct      = 40 + int((epoch + 1) / self.total_epochs * 45)
                val_acc  = logs.get('val_accuracy', 0.0)
                train_acc = logs.get('accuracy', 0.0)
                _set(
                    progress=pct,
                    message=(
                        f'Epoch {epoch + 1}/{self.total_epochs} — '
                        f'train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}'
                    ),
                )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[
                _ProgressCallback(EPOCHS),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=3, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=2, verbose=0
                ),
            ],
        )

        # ── 7. Evaluate ────────────────────────────────────────────────────────
        _set(phase='evaluating', progress=87, message='Evaluating on validation set…')
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        final_acc = round(float(val_acc) * 100, 2)
        _set(
            progress=93,
            accuracy=final_acc,
            message=f'Evaluation done — val_accuracy: {val_acc:.4f} ({final_acc}%)',
        )

        # ── 8. Save ────────────────────────────────────────────────────────────
        _set(phase='saving', progress=96, message='Saving retrained model…')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)

        output_filename = os.path.basename(output_path)
        _set(
            phase='done',
            progress=100,
            is_running=False,
            finished_at=datetime.datetime.now().isoformat(),
            output_filename=output_filename,
            message=f'Retraining complete. Model saved as "{output_filename}".',
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set(
            phase='error',
            is_running=False,
            finished_at=datetime.datetime.now().isoformat(),
            error=str(exc),
            message=f'Retraining failed: {exc}',
        )

    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── public entry point ─────────────────────────────────────────────────────────

def start_retraining(model_type: str, base_model_path: str, output_path: str) -> bool:
    """
    Kick off retraining in a background daemon thread.

    Returns True if the job was started, False if one is already running.
    """
    with _lock:
        if _status['is_running']:
            return False
        _status.update({
            'is_running':      True,
            'model_type':      model_type,
            'phase':           'starting',
            'progress':        0,
            'message':         'Initialising retraining job…',
            'started_at':      datetime.datetime.now().isoformat(),
            'finished_at':     None,
            'output_filename': None,
            'accuracy':        None,
            'error':           None,
            'dataset_info':    None,
        })

    thread = threading.Thread(
        target=_run_retraining,
        args=(model_type, base_model_path, output_path),
        daemon=True,
        name=f'mangosense-retrain-{model_type}',
    )
    thread.start()
    return True
