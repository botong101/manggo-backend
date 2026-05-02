import datetime
import os
import shutil
import tempfile
import threading

from .config import RetrainConfig
from .dataset import build_temp_dataset, collect_verified_images
from .state import _lock, _set, _status


def _run_retraining(model_type: str, base_model_path: str, output_path: str, config: RetrainConfig) -> None:
    tmp_dir = None
    try:
        import tensorflow as tf

        # ── 1. Download verified + training_ready images to local cache ────────
        _set(
            phase='downloading',
            progress=5,
            message='Downloading verified + training_ready images from storage…',
        )
        class_map = collect_verified_images(model_type, config.min_images_per_class)

        if len(class_map) < 2:
            raise ValueError(
                f"Not enough eligible classes after download. "
                f"Need at least 2 classes with {config.min_images_per_class}+ images "
                f"(is_verified=True AND training_ready=True). "
                f"Found: {list(class_map.keys()) or 'none'}"
            )

        total_images = sum(len(v) for v in class_map.values())
        _set(
            progress=10,
            message=f'Found {total_images} images across {len(class_map)} classes: {list(class_map.keys())}',
        )

        # ── 2. Build train/val directory split ─────────────────────────────────
        _set(phase='preparing', progress=15, message='Preparing dataset directory…')
        tmp_dir = tempfile.mkdtemp(prefix='mangosense_retrain_')
        train_dir, val_dir, dataset_info = build_temp_dataset(class_map, tmp_dir, config.val_split)
        _set(progress=20, dataset_info=dataset_info, message='Dataset directory ready.')

        # ── 3. Create tf.data datasets ─────────────────────────────────────────
        _set(phase='training', progress=25, message='Loading TF datasets…')
        IMG_SIZE = (224, 224)

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=config.batch_size,
            label_mode='categorical',
            shuffle=True,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=config.batch_size,
            label_mode='categorical',
        )

        num_classes = len(train_ds.class_names)
        _set(progress=30, message=f'Classes ({num_classes}): {train_ds.class_names}')

        # ── 4. Load base model or build fresh MobileNetV2 ─────────────────────
        _set(progress=35, message='Loading base model…')

        if os.path.exists(base_model_path):
            model = tf.keras.models.load_model(base_model_path)
            if model.output_shape[-1] != num_classes:
                _set(message=(
                    f'Output class count changed ({model.output_shape[-1]} → {num_classes}). '
                    'Rebuilding classification head…'
                ))
                x = model.layers[-2].output
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='retrain_predictions')(x)
                model = tf.keras.Model(inputs=model.input, outputs=outputs)
        else:
            _set(message='Base model not found. Building new MobileNetV2 model…')
            data_aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
            base = tf.keras.applications.MobileNetV2(
                include_top=False, input_shape=(224, 224, 3), weights='imagenet', pooling='avg',
            )
            base.trainable = False
            inputs  = tf.keras.Input(shape=(224, 224, 3))
            x       = data_aug(inputs)
            x       = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
            x       = base(x, training=False)
            x       = tf.keras.layers.Dense(128, activation='relu')(x)
            x       = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model   = tf.keras.Model(inputs, outputs)

        # ── 5. Freeze base; unfreeze top N layers for fine-tuning ─────────────
        total_layers = len(model.layers)
        freeze_until = max(0, total_layers - config.unfreeze_top_n_layers)
        for i, layer in enumerate(model.layers):
            layer.trainable = i >= freeze_until

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        _set(progress=40, message='Model compiled. Starting fine-tuning…')

        # ── 6. Train with callbacks ────────────────────────────────────────────
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
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            callbacks=[
                _ProgressCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=config.early_stopping_patience, restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=config.lr_reduce_factor, patience=config.lr_reduce_patience, verbose=0,
                ),
            ],
        )

        # ── 7. Evaluate ────────────────────────────────────────────────────────
        _set(phase='evaluating', progress=87, message='Evaluating on validation set…')
        _, val_acc = model.evaluate(val_ds, verbose=0)
        final_acc = round(float(val_acc) * 100, 2)
        _set(progress=93, accuracy=final_acc, message=f'Evaluation done — val_accuracy: {val_acc:.4f} ({final_acc}%)')

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


def start_retraining(
    model_type: str,
    base_model_path: str,
    output_path: str,
    config: RetrainConfig | None = None,
    model_kind: str = 'mobilenetv2',
) -> bool:
    """
    Kick off retraining in a background daemon thread.
    Returns True if started, False if a job is already running.
    """
    cfg = config or RetrainConfig()

    with _lock:
        if _status['is_running']:
            return False
        _status.update({
            'is_running':      True,
            'model_type':      model_type,
            'model_kind':      model_kind,
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

    threading.Thread(
        target=_run_retraining,
        args=(model_type, base_model_path, output_path, cfg),
        daemon=True,
        name=f'mangosense-retrain-{model_type}',
    ).start()
    return True
