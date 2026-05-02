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
    'model_variant':   None,    # 'standard' | 'hybrid'
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
        {class_label: [(absolute_file_path, MangoImage_id), ...]}
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
            class_map.setdefault(label, []).append((abs_path, img.id))

    return {k: v for k, v in class_map.items() if len(v) >= MIN_IMAGES_PER_CLASS}


def get_dataset_preview(model_type: str) -> dict:
    """
    Return a preview of available verified images and symptom data without starting training.
    Used by the dataset-info endpoint.
    """
    from ..models import MangoImage
    import json

    # all verified images for this type (regardless of min threshold)
    qs = MangoImage.objects.filter(
        is_verified=True,
        disease_type=model_type,
    ).exclude(disease_classification='').exclude(disease_classification__isnull=True)

    all_counts: dict = {}
    symptom_stats: dict = {}
    all_unique_symptoms = set()
    media_root = settings.MEDIA_ROOT

    for img in qs:
        label = img.disease_classification.strip()
        if not label:
            continue
        abs_path = os.path.join(media_root, str(img.image))
        if os.path.isfile(abs_path):
            all_counts[label] = all_counts.get(label, 0) + 1
            
            # Initialize symptom stats for this class if not exists
            if label not in symptom_stats:
                symptom_stats[label] = {
                    'images_with_symptoms': 0,
                    'symptom_mentions': 0,
                    'unique_symptoms': set(),
                }
            
            # Collect symptoms from this image
            symptoms_list = []
            if img.primary_symptoms:
                try:
                    if isinstance(img.primary_symptoms, str):
                        symptoms_list.extend(json.loads(img.primary_symptoms))
                    else:
                        symptoms_list.extend(img.primary_symptoms)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            if not symptoms_list and img.selected_symptoms:
                try:
                    if isinstance(img.selected_symptoms, str):
                        symptoms_list.extend(json.loads(img.selected_symptoms))
                    else:
                        symptoms_list.extend(img.selected_symptoms)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            if symptoms_list:
                symptom_stats[label]['images_with_symptoms'] += 1
                symptom_stats[label]['symptom_mentions'] += len(symptoms_list)
                symptom_stats[label]['unique_symptoms'].update(symptoms_list)
                all_unique_symptoms.update(symptoms_list)

    eligible = {k: v for k, v in all_counts.items() if v >= MIN_IMAGES_PER_CLASS}
    can_retrain = len(eligible) >= 2

    # Convert sets to counts for JSON serialization
    symptom_stats_serializable = {}
    for label, stats in symptom_stats.items():
        symptom_stats_serializable[label] = {
            'image_count': all_counts[label],
            'images_with_symptoms': stats['images_with_symptoms'],
            'symptom_mentions': stats['symptom_mentions'],
            'unique_symptoms': len(stats['unique_symptoms']),
            'avg_symptoms_per_image': round(
                stats['symptom_mentions'] / stats['images_with_symptoms']
                if stats['images_with_symptoms'] > 0 else 0, 2
            ),
        }

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
        'symptom_stats': {
            'total_unique_symptoms': len(all_unique_symptoms),
            'per_class': symptom_stats_serializable,
        },
    }


# ── temp dataset builder ───────────────────────────────────────────────────────

def _build_temp_dataset(class_map: dict, tmp_dir: str, val_split: float = 0.2):
    """
    Copies verified images into a Keras-ready directory structure:

        tmp_dir/train/<class>/<image>
        tmp_dir/val/<class>/<image>

    Returns (train_dir, val_dir, dataset_info, image_to_db_id)
    where image_to_db_id maps: 'path/to/train/class/img.jpg' -> MangoImage.id
    """
    train_dir = os.path.join(tmp_dir, 'train')
    val_dir   = os.path.join(tmp_dir, 'val')
    dataset_info = {}
    image_to_db_id = {}  # Mapping from copied image path to MangoImage ID

    for label, path_id_pairs in class_map.items():
        shuffled = path_id_pairs[:]
        random.shuffle(shuffled)

        n_val   = max(1, int(len(shuffled) * val_split))
        n_train = len(shuffled) - n_val

        splits = [('train', shuffled[:n_train]), ('val', shuffled[n_train:])]

        for split_name, split_path_ids in splits:
            dest_dir = os.path.join(tmp_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            for src, db_id in split_path_ids:
                base, ext = os.path.splitext(os.path.basename(src))
                dst_filename = f"{base}_{id(src)}{ext}"
                dst = os.path.join(dest_dir, dst_filename)
                shutil.copy2(src, dst)
                
                # Map relative path to database ID
                rel_path = os.path.relpath(dst, tmp_dir)
                image_to_db_id[rel_path] = db_id

        dataset_info[label] = {
            'total': len(shuffled),
            'train': n_train,
            'val':   len(shuffled) - n_train,
        }

    return train_dir, val_dir, dataset_info, image_to_db_id


# ── training ───────────────────────────────────────────────────────────────────

# ── hybrid model helpers ───────────────────────────────────────────────────────

def _collect_binary_symptom_data(model_type: str, class_names: list, image_to_db_id: dict):
    """
    Build binary (0/1) symptom vectors from the DiseaseSymptom DB table.

    Vocabulary is drawn from Symptom rows ordered by (vector_index, key) so
    indices are stable and consistent with any existing prototype sidecar.

    Returns
    -------
    vocab             : list[str]  — symptom keys in index order
    num_features      : int
    disease_vectors   : {class_name: np.float32 binary array}  — from DiseaseSymptom table
    per_image_vectors : {rel_path:  np.float32 binary array}
        Uses MangoImage.selected_symptoms when present, else the disease prototype.
    """
    from ..models import Symptom, Disease, DiseaseSymptom, MangoImage
    import json

    # ── build vocabulary from Symptom table ───────────────────────────────
    vocab = list(
        Symptom.objects.filter(plant_part=model_type, is_in_vocabulary=True)
        .order_by('vector_index', 'key')
        .values_list('key', flat=True)
    )
    if not vocab:
        _set(message='WARNING: No symptoms in vocabulary — using 8-dim zero vectors.')
    num_features = max(len(vocab), 8)
    vocab_index  = {k: i for i, k in enumerate(vocab)}

    # ── binary disease prototype vectors (from DiseaseSymptom table) ───────
    disease_vectors: dict = {}
    for cls in class_names:
        try:
            disease = Disease.objects.get(name=cls, plant_part=model_type)
            vec = np.zeros(num_features, dtype=np.float32)
            for link in DiseaseSymptom.objects.filter(disease=disease).select_related('symptom'):
                idx = vocab_index.get(link.symptom.key)
                if idx is not None:
                    vec[idx] = 1.0
            disease_vectors[cls] = vec
        except Disease.DoesNotExist:
            disease_vectors[cls] = np.zeros(num_features, dtype=np.float32)

    _set(message=(
        f'Symptom vocab: {len(vocab)} keys. '
        f'Prototypes built for: {list(disease_vectors.keys())}'
    ))

    # ── per-image vectors ──────────────────────────────────────────────────
    # rel_path looks like "train\Anthracnose\img_xxx.jpg" (or with forward slashes)
    db_ids        = set(image_to_db_id.values())
    image_records = {img.id: img for img in MangoImage.objects.filter(id__in=db_ids)}
    per_image_vectors: dict = {}

    for rel_path, db_id in image_to_db_id.items():
        # Derive class name from second path segment (after split/train/val prefix)
        parts      = rel_path.replace('\\', '/').split('/')
        class_name = parts[1] if len(parts) >= 3 else None
        default    = disease_vectors.get(class_name, np.zeros(num_features, dtype=np.float32))

        img_record = image_records.get(db_id)
        if img_record and img_record.selected_symptoms:
            try:
                syms = img_record.selected_symptoms
                if isinstance(syms, str):
                    syms = json.loads(syms)
                if syms:
                    real_vec = np.zeros(num_features, dtype=np.float32)
                    for key in syms:
                        idx = vocab_index.get(key)
                        if idx is not None:
                            real_vec[idx] = 1.0
                    per_image_vectors[rel_path] = real_vec
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

        per_image_vectors[rel_path] = default.copy()

    return vocab, num_features, disease_vectors, per_image_vectors


def _conv_block(x, filters, tf):
    """Single conv→BN→ReLU→MaxPool block with L2 regularisation."""
    reg = tf.keras.regularizers.l2(1e-4)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    return x


def _build_hybrid_model(input_shape: tuple, num_symptom_features: int, num_classes: int):
    """
    Exact replica of MangoSenseNet-CoAttn from train-custom-cnn-symptom.py.
    input_shape should be (240, 240, 3).
    """
    import tensorflow as tf

    TOKEN_DIM  = 16
    IMG_TOKENS = 16
    SYM_TOKENS = 4
    NUM_HEADS  = 4
    KEY_DIM    = TOKEN_DIM // NUM_HEADS  # 4

    # ── image branch ──────────────────────────────────────────────────────
    img_input = tf.keras.Input(shape=input_shape, name='image')
    x = tf.keras.layers.RandomFlip('horizontal')(img_input)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.15)(x)
    x = tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)
    x = tf.keras.layers.RandomBrightness(0.2)(x)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = _conv_block(x, 32,  tf)
    x = _conv_block(x, 64,  tf)
    x = _conv_block(x, 128, tf)
    x = _conv_block(x, 256, tf)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(IMG_TOKENS * TOKEN_DIM, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    img_tokens = tf.keras.layers.Reshape((IMG_TOKENS, TOKEN_DIM))(x)  # (B, 16, 16)

    # ── symptom branch ────────────────────────────────────────────────────
    sym_input = tf.keras.Input(shape=(num_symptom_features,), name='symptoms')
    s = tf.keras.layers.BatchNormalization()(sym_input)
    s = tf.keras.layers.Dense(SYM_TOKENS * TOKEN_DIM, activation='relu')(s)
    s = tf.keras.layers.Dropout(0.3)(s)
    sym_tokens = tf.keras.layers.Reshape((SYM_TOKENS, TOKEN_DIM))(s)  # (B, 4, 16)

    # ── bidirectional co-attention ────────────────────────────────────────
    img_attended = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
        query=img_tokens, key=sym_tokens, value=sym_tokens
    )
    sym_attended = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
        query=sym_tokens, key=img_tokens, value=img_tokens
    )
    img_out = tf.keras.layers.LayerNormalization()(img_tokens + img_attended)
    sym_out = tf.keras.layers.LayerNormalization()(sym_tokens + sym_attended)

    img_flat = tf.keras.layers.Flatten()(img_out)
    img_flat = tf.keras.layers.Dropout(0.3)(img_flat)
    sym_flat = tf.keras.layers.Flatten()(sym_out)
    sym_flat = tf.keras.layers.Dropout(0.3)(sym_flat)

    fused   = tf.keras.layers.Concatenate()([img_flat, sym_flat])
    out     = tf.keras.layers.Dense(128, activation='relu')(fused)
    out     = tf.keras.layers.Dropout(0.4)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(out)

    return tf.keras.Model(
        inputs={'image': img_input, 'symptoms': sym_input},
        outputs=outputs,
        name='MangoSenseNet_CoAttn',
    )


def _run_retraining(model_type: str, base_model_path: str, output_path: str, model_variant: str = 'standard'):
    """
    Full retraining pipeline — runs inside a background thread.
    Updates module-level _status at every major step.
    
    model_variant: 'standard' (MobileNetV2) or 'hybrid' (MangoSenseNet-CoAttn)
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
        train_dir, val_dir, dataset_info, image_to_db_id = _build_temp_dataset(class_map, tmp_dir)
        _set(progress=20, dataset_info=dataset_info, message='Dataset directory ready.')

        # ── 3b. For hybrid models: build binary symptom vectors from DiseaseSymptom ──
        image_symptoms       = None
        num_symptom_features = 0
        disease_vectors: dict = {}

        if model_variant == 'hybrid':
            _set(message='Building binary symptom vocabulary from DiseaseSymptom table…')
            _, num_symptom_features, disease_vectors, image_symptoms = _collect_binary_symptom_data(
                model_type, list(class_map.keys()), image_to_db_id
            )

        # ── 3. Create tf.data datasets ────────────────────────────────────────
        _set(phase='training', progress=25, message='Loading TF datasets…')
        # Hybrid uses 240×240 to match the original CoAttn training script
        IMG_SIZE         = (240, 240) if model_variant == 'hybrid' else (224, 224)
        BATCH_SIZE       = 16
        # Whole-vector modality dropout rate (p=0 → always provide symptoms)
        MODALITY_DROPOUT = 0.5 if model_type == 'leaf' else 0.2

        if model_variant == 'hybrid':
            _set(message='Building hybrid dataset with binary symptom vectors…')

            class_index: dict = {}
            train_files: list = []
            val_files:   list = []

            for class_name in sorted(os.listdir(train_dir)):
                class_dir = os.path.join(train_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                class_idx = len(class_index)
                class_index[class_name] = class_idx
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if os.path.isfile(fpath):
                        rel_path = os.path.relpath(fpath, tmp_dir)
                        train_files.append((fpath, class_idx, rel_path))

            for class_name in sorted(os.listdir(val_dir)):
                class_dir = os.path.join(val_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                class_idx = class_index.get(class_name, 0)
                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if os.path.isfile(fpath):
                        rel_path = os.path.relpath(fpath, tmp_dir)
                        val_files.append((fpath, class_idx, rel_path))

            random.shuffle(train_files)

            def _load_img(fpath):
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=IMG_SIZE)
                return tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)

            # Training set — apply modality dropout (whole-vector zeroing)
            train_images, train_labels, train_symptoms = [], [], []
            zero_sym = np.zeros(num_symptom_features, dtype=np.float32)
            for fpath, class_idx, rel_path in train_files:
                try:
                    img_arr = _load_img(fpath)
                    sym_vec = image_symptoms.get(rel_path, zero_sym).copy()
                    if random.random() < MODALITY_DROPOUT:
                        sym_vec = zero_sym.copy()
                    train_images.append(img_arr)
                    train_labels.append(class_idx)
                    train_symptoms.append(sym_vec)
                except Exception as e:
                    _set(message=f'WARNING: Could not load image {fpath}: {e}')

            train_images_np   = np.array(train_images,   dtype=np.float32)
            train_labels_cat  = tf.keras.utils.to_categorical(train_labels, num_classes=len(class_index))
            train_symptoms_np = np.array(train_symptoms, dtype=np.float32)

            train_ds = tf.data.Dataset.from_tensor_slices((
                {'image': train_images_np, 'symptoms': train_symptoms_np},
                train_labels_cat,
            )).shuffle(len(train_images_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            # Validation set — oracle mode: always provide the disease prototype
            val_images, val_labels, val_symptoms = [], [], []
            for fpath, class_idx, rel_path in val_files:
                try:
                    img_arr = _load_img(fpath)
                    sym_vec = image_symptoms.get(rel_path, zero_sym)
                    val_images.append(img_arr)
                    val_labels.append(class_idx)
                    val_symptoms.append(sym_vec)
                except Exception as e:
                    _set(message=f'WARNING: Could not load validation image {fpath}: {e}')

            val_images_np   = np.array(val_images,   dtype=np.float32)
            val_labels_cat  = tf.keras.utils.to_categorical(val_labels, num_classes=len(class_index))
            val_symptoms_np = np.array(val_symptoms, dtype=np.float32)

            val_ds = tf.data.Dataset.from_tensor_slices((
                {'image': val_images_np, 'symptoms': val_symptoms_np},
                val_labels_cat,
            )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            class_names = sorted(class_index.keys())
            num_classes = len(class_names)
        else:
            # Standard: use image_dataset_from_directory
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

        num_classes = len(train_ds.class_names) if model_variant != 'hybrid' else num_classes
        class_names = train_ds.class_names if model_variant != 'hybrid' else class_names
        _set(progress=30, message=f'Classes ({num_classes}): {class_names}')

        # ── 4. Load or build model ────────────────────────────────────────────
        _set(progress=35, message='Loading base model…')

        if model_variant == 'hybrid':
            # Always build fresh — preserves correct input shape for binary symptom vectors
            _set(message='Building new MangoSenseNet-CoAttn hybrid model…')
            model = _build_hybrid_model((240, 240, 3), num_symptom_features, num_classes)
        elif os.path.exists(base_model_path):
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
            x       = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
            x       = base(x, training=False)
            x       = tf.keras.layers.Dense(128, activation='relu')(x)
            x       = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model   = tf.keras.Model(inputs, outputs)

        # ── 5. Freeze / compile ───────────────────────────────────────────────
        if model_variant == 'hybrid':
            # Fresh build — train all layers from scratch with a higher LR
            for layer in model.layers:
                layer.trainable = True
            lr     = 1e-3
            EPOCHS = 30
        else:
            # Fine-tune: freeze all but the top 20 layers
            total_layers = len(model.layers)
            freeze_until = max(0, total_layers - 20)
            for i, layer in enumerate(model.layers):
                layer.trainable = i >= freeze_until
            lr     = 1e-4
            EPOCHS = 10

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        _set(progress=40, message='Model compiled. Starting training…')

        # ── 6. Training callbacks ──────────────────────────────────────────────

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
                    monitor='val_loss',
                    patience=5 if model_variant == 'hybrid' else 3,
                    restore_best_weights=True,
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

        # Save binary prototype sidecar alongside the hybrid model so ml_views.py
        # picks it up automatically for two-pass inference.
        if model_variant == 'hybrid' and disease_vectors:
            import json as _json
            proto_path = output_path.replace('.keras', '_prototypes.json')
            prototypes = {
                name: disease_vectors[name].tolist()
                for name in class_names
                if name in disease_vectors
            }
            with open(proto_path, 'w') as _pf:
                _json.dump(prototypes, _pf, indent=2)
            _set(message=f'Prototype sidecar saved: {os.path.basename(proto_path)}')

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

def start_retraining(model_type: str, base_model_path: str, output_path: str, model_variant: str = 'standard') -> bool:
    """
    Kick off retraining in a background daemon thread.
    
    Args:
        model_type: 'leaf' or 'fruit'
        base_model_path: Path to existing model (ignored for hybrid)
        output_path: Where to save the retrained model
        model_variant: 'standard' (MobileNetV2) or 'hybrid' (MangoSenseNet-CoAttn)

    Returns True if the job was started, False if one is already running.
    """
    with _lock:
        if _status['is_running']:
            return False
        _status.update({
            'is_running':      True,
            'model_type':      model_type,
            'model_variant':   model_variant,
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
        args=(model_type, base_model_path, output_path, model_variant),
        daemon=True,
        name=f'mangosense-retrain-{model_type}-{model_variant}',
    )
    thread.start()
    return True
