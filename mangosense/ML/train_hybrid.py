import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ============================================================
# Simple utility: make path join robust for Colab vs local
# ============================================================
def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

if _is_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/MangoSense'
else:
    BASE_DIR = ''

def p(*parts):
    return os.path.join(BASE_DIR, *parts) if BASE_DIR else os.path.join(*parts)


# ============================================================
# CONFIG (adjust paths as needed)
# ============================================================
SYMPTOM_CSV = p('symptom_data', 'all_datasets_symptom_features.csv')

META_COLUMNS = {
    'dataset', 'split', 'class', 'filename', 'filepath', 'relative_path', 'color_dominant_hue_bin'
}

DATASETS = {
    'leaves': {
        'dataset_filter': 'Leaves_Mixed',
        'data_dir': p('datasets', 'preprocessed-leaves-dual'),
        'save_dir': p('models-custom-hybrid-leaves'),
        'img_size': (240, 240),
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'modality_dropout': 0.5,
    },
    'fruit': {
        'dataset_filter': 'Mango_Fruits',
        'data_dir': p('datasets', 'preprocessed-fruit-dual'),
        'save_dir': p('models-custom-hybrid-fruit'),
        'img_size': (240, 240),
        'num_epochs': 60,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'modality_dropout': 0.2,
    },
}


# ============================================================
# Model builder (unchanged architecture)
# ============================================================
_REG = tf.keras.regularizers.l2(1e-4)


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=_REG)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    return x


def build_hybrid_model(input_shape, num_symptom_features, num_classes):
    TOKEN_DIM = 16
    IMG_TOKENS = 16
    SYM_TOKENS = 4
    NUM_HEADS = 4
    KEY_DIM = TOKEN_DIM // NUM_HEADS

    img_input = tf.keras.Input(shape=input_shape, name='image')
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ], name='augmentation')

    x = augmentation(img_input)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = tf.keras.layers.GlobalAveragePooling2D(name='image_gap')(x)
    x = tf.keras.layers.Dense(IMG_TOKENS * TOKEN_DIM, activation='relu', kernel_regularizer=_REG)(x)
    x = tf.keras.layers.Dropout(0.5, name='img_dropout')(x)
    img_tokens = tf.keras.layers.Reshape((IMG_TOKENS, TOKEN_DIM), name='img_tokens')(x)

    sym_input = tf.keras.Input(shape=(num_symptom_features,), name='symptoms')
    s = tf.keras.layers.BatchNormalization()(sym_input)
    s = tf.keras.layers.Dense(SYM_TOKENS * TOKEN_DIM, activation='relu', kernel_regularizer=_REG)(s)
    s = tf.keras.layers.Dropout(0.3, name='sym_dropout')(s)
    sym_tokens = tf.keras.layers.Reshape((SYM_TOKENS, TOKEN_DIM), name='sym_tokens')(s)

    img_attended = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, name='img_attends_sym')(
        query=img_tokens, key=sym_tokens, value=sym_tokens
    )
    sym_attended = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, name='sym_attends_img')(
        query=sym_tokens, key=img_tokens, value=img_tokens
    )

    img_out = tf.keras.layers.LayerNormalization(name='img_layernorm')(img_tokens + img_attended)
    sym_out = tf.keras.layers.LayerNormalization(name='sym_layernorm')(sym_tokens + sym_attended)

    img_flat = tf.keras.layers.Dropout(0.3)(tf.keras.layers.Flatten(name='img_flat')(img_out))
    sym_flat = tf.keras.layers.Dropout(0.3)(tf.keras.layers.Flatten(name='sym_flat')(sym_out))
    fused = tf.keras.layers.Concatenate(name='fusion')([img_flat, sym_flat])
    out = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=_REG)(fused)
    out = tf.keras.layers.Dropout(0.4)(out)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(out)

    return tf.keras.Model(inputs={'image': img_input, 'symptoms': sym_input}, outputs=outputs, name='MangoSenseNet_CoAttn')


# ============================================================
# Symptom feature helpers (fixed/robust)
# ============================================================
def compute_class_means(df, feature_cols):
    means = df.groupby('class')[feature_cols].mean()
    return means


def build_ordered_class_prototypes(means_df, class_names, feature_cols):
    """
    Ensure prototypes are ordered to match dataset `class_names`.
    Returns (num_classes, num_features) numpy array and a dict map.
    """
    prototypes = []
    missing = []
    for c in class_names:
        if c in means_df.index:
            prototypes.append(means_df.loc[c].values.astype(np.float32))
        else:
            prototypes.append(np.zeros(len(feature_cols), dtype=np.float32))
            missing.append(c)
    if missing:
        print(f"WARNING: No prototype rows for classes: {missing}")
    return np.vstack(prototypes)


def attach_class_symptoms(class_feat_tensor, training=False, dropout_rate=0.5):
    """
    Returns a tf.data map fn that attaches class-prototype symptom features.
    Robust: ensures dtype and shape match model expectations.
    """
    def _fn(image, label):
        class_idx = tf.argmax(label, axis=-1)
        symptoms = tf.gather(class_feat_tensor, class_idx)
        symptoms = tf.cast(symptoms, tf.float32)
        if training and dropout_rate > 0.0:
            batch_size = tf.shape(symptoms)[0]
            keep = tf.cast(tf.random.uniform([batch_size, 1]) > dropout_rate, tf.float32)
            symptoms = symptoms * keep  # broadcast across feature dim
        return {'image': image, 'symptoms': symptoms}, label
    return _fn


def zero_symptoms_fn(num_features):
    def _fn(image, label):
        zeros = tf.zeros((tf.shape(image)[0], num_features), dtype=tf.float32)
        return {'image': image, 'symptoms': zeros}, label
    return _fn


# ============================================================
# Training loop with explicit prototype ordering and checks
# ============================================================
all_results = {}

print(f"Loading symptom CSV: {SYMPTOM_CSV}")
df_all = pd.read_csv(SYMPTOM_CSV)
print(f"Total rows: {len(df_all)}")
print(f"Datasets found: {df_all['dataset'].unique().tolist()}")

for dataset, cfg in DATASETS.items():
    dataset_filter = cfg['dataset_filter']
    data_dir = cfg['data_dir']
    save_dir = cfg['save_dir']
    img_size = cfg['img_size']
    num_epochs = cfg['num_epochs']
    learning_rate = cfg['learning_rate']
    batch_size = cfg['batch_size']
    modality_dropout = cfg['modality_dropout']
    input_shape = (*img_size, 3)

    print('\n' + '='*55)
    print(f"  Training MangoSenseNet-CoAttn — {dataset.upper()}")
    print('='*55)
    print(f"  LR: {learning_rate}  |  Batch: {batch_size}  |  Epochs: {num_epochs}  |  Modality dropout: {modality_dropout}")
    print(f"  Image dir: {data_dir}")

    if not os.path.isdir(data_dir):
        print(f"  ERROR: Image directory not found: {data_dir}")
        continue

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'), image_size=img_size, batch_size=batch_size,
        label_mode='categorical', shuffle=True, seed=42
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'), image_size=img_size, batch_size=batch_size,
        label_mode='categorical', shuffle=False
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'test'), image_size=img_size, batch_size=batch_size,
        label_mode='categorical', shuffle=False
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    # Compute class-level prototypes from symptom CSV and align to class_names
    df = df_all[df_all['dataset'] == dataset_filter].copy()
    if df.empty:
        print(f"  WARNING: No CSV rows for '{dataset_filter}'. Skipping.")
        continue

    feature_cols = [c for c in df.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=feature_cols)
    print(f"  Symptom features: {len(feature_cols)}")

    means_df = compute_class_means(df, feature_cols)
    ordered_prototypes = build_ordered_class_prototypes(means_df, class_names, feature_cols)

    scaler = StandardScaler()
    class_means_scaled = scaler.fit_transform(ordered_prototypes).astype(np.float32)
    class_feat_tensor = tf.constant(class_means_scaled)

    print('  Class prototype norms:')
    for i, c in enumerate(class_names):
        print(f"    {c:<22} norm={np.linalg.norm(class_means_scaled[i]):.3f}")

    num_features = len(feature_cols)
    train_attach = attach_class_symptoms(class_feat_tensor, training=True, dropout_rate=modality_dropout)
    oracle_attach = attach_class_symptoms(class_feat_tensor, training=False)
    zero_attach = zero_symptoms_fn(num_features)

    train_ds = (
        train_ds_raw.cache().shuffle(1000, seed=42)
        .map(train_attach, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw.cache()
        .map(oracle_attach, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    )

    model = build_hybrid_model(input_shape, num_features, num_classes)
    model.summary()
    print(f"  Total parameters: {model.count_params():,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(save_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'best_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=callbacks, verbose=1)

    # Evaluation: zero-symptom (image-only baseline) and oracle (prototype attached)
    test_zero = test_ds_raw.cache().map(zero_attach, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    loss_zero, acc_zero = model.evaluate(test_zero, verbose=0)
    print(f"\n  [A] Zero-symptom (image-only baseline):  Acc={acc_zero:.4f}  Loss={loss_zero:.4f}")

    y_pred_zero = np.argmax(model.predict(test_zero, verbose=0), axis=1)
    y_true = np.argmax(np.concatenate([y.numpy() for _, y in test_zero]), axis=1)

    test_oracle = test_ds_raw.cache().map(oracle_attach, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    loss_oracle, acc_oracle = model.evaluate(test_oracle, verbose=0)
    print(f"  [B] Oracle symptoms (upper bound):       Acc={acc_oracle:.4f}  Loss={loss_oracle:.4f}")
    print(f"      Symptom benefit: +{(acc_oracle - acc_zero)*100:.1f}% vs image-only")

    report = classification_report(y_true, y_pred_zero, target_names=class_names, digits=4)
    print('\nClassification Report (zero-symptom / image-only baseline):')
    print(report)

    # Save artifacts
    model.save(os.path.join(save_dir, 'hybrid_cnn.keras'))
    joblib.dump(scaler, os.path.join(save_dir, 'symptom_scaler.joblib'))
    with open(os.path.join(save_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f)
    class_proto = {c: class_means_scaled[i].tolist() for i, c in enumerate(class_names)}
    with open(os.path.join(save_dir, 'class_symptom_prototypes.json'), 'w') as f:
        json.dump(class_proto, f)
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Accuracy (zero-symptom / image-only): {acc_zero:.4f}\n")
        f.write(f"Accuracy (oracle / upper bound):      {acc_oracle:.4f}\n")
        f.write(f"Loss     (zero-symptom):              {loss_zero:.4f}\n\n")
        f.write(report)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'Accuracy — {dataset}')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'Loss — {dataset}')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_zero)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix — MangoSenseNet-CoAttn ({dataset}, zero-symptom)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)

    all_results[dataset] = {'acc_zero': round(acc_zero, 4), 'acc_oracle': round(acc_oracle, 4), 'loss_zero': round(loss_zero, 4)}
    print(f"  Results saved to: {save_dir}")

print('\n' + '='*55)
print('  FINAL RESULTS — MangoSenseNet-CoAttn')
print('='*55)
print(f"  {'Dataset':<10} {'Image-only (A)':<18} {'Oracle / +Symptoms (B)'}")
print('  ' + '-'*52)
for dataset, res in all_results.items():
    gain = (res['acc_oracle'] - res['acc_zero']) * 100
    print(f"  {dataset:<10} {res['acc_zero']:.4f}             {res['acc_oracle']:.4f}  (+{gain:.1f}%)")
