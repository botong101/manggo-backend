import os
import random
import shutil

from django.conf import settings

from .cache import download_image_to_cache, RETRAIN_CACHE_DIR
from .config import MIN_IMAGES_PER_CLASS
from .state import _set

_PREPROC_DIR = os.path.join(settings.MEDIA_ROOT, 'retrain_preprocessed')


def _preprocessed_path(raw_cache_path: str) -> str:
    """Return the preprocessed counterpart for a raw cache file."""
    rel  = os.path.relpath(raw_cache_path, RETRAIN_CACHE_DIR)
    base = os.path.splitext(rel)[0]
    return os.path.join(_PREPROC_DIR, base + '.png')


def collect_verified_images(model_type: str, min_images_per_class: int = MIN_IMAGES_PER_CLASS) -> dict:
    """
    Query MangoImage rows that are both is_verified=True and training_ready=True,
    download each to the persistent retrain cache, and return a class→path map.

    Returns:
        {class_label: [absolute_local_path, ...]}
    Only classes with >= min_images_per_class successfully cached files included.
    """
    from ...models import MangoImage

    qs = MangoImage.objects.filter(
        is_verified=True,
        training_ready=True,
        disease_type=model_type,
    ).exclude(disease_classification='').exclude(disease_classification__isnull=True)

    total = qs.count()
    if total == 0:
        return {}

    class_map: dict = {}
    downloaded = 0
    failed = 0

    for img in qs:
        label = (img.disease_classification or '').strip()
        if not label:
            continue

        local_path = download_image_to_cache(img, model_type)
        if local_path and os.path.isfile(local_path):
            preproc = _preprocessed_path(local_path)
            final   = preproc if os.path.isfile(preproc) else local_path
            class_map.setdefault(label, []).append(final)
            downloaded += 1
        else:
            failed += 1

        pct = 5 + int(((downloaded + failed) / total) * 10)
        _set(
            phase='downloading',
            progress=pct,
            message=(
                f'Downloading verified images… {downloaded + failed}/{total} '
                f'(ok: {downloaded}, failed: {failed})'
            ),
        )

    return {k: v for k, v in class_map.items() if len(v) >= min_images_per_class}


def get_dataset_preview(model_type: str, min_images_per_class: int = MIN_IMAGES_PER_CLASS) -> dict:
    """
    Return per-class image counts for the dataset-info endpoint.
    Does NOT download files — uses storage.exists() only.
    """
    from ...models import MangoImage

    qs = MangoImage.objects.filter(
        is_verified=True,
        training_ready=True,
        disease_type=model_type,
    ).exclude(disease_classification='').exclude(disease_classification__isnull=True)

    all_counts: dict = {}

    for img in qs:
        label = (img.disease_classification or '').strip()
        if not label:
            continue
        try:
            if img.image and img.image.name and img.image.storage.exists(img.image.name):
                all_counts[label] = all_counts.get(label, 0) + 1
        except Exception:
            continue

    eligible = {k: v for k, v in all_counts.items() if v >= min_images_per_class}
    can_retrain = len(eligible) >= 2

    return {
        'model_type':             model_type,
        'all_classes':            all_counts,
        'eligible_classes':       eligible,
        'total_eligible_images':  sum(eligible.values()),
        'min_images_per_class':   min_images_per_class,
        'can_retrain':            can_retrain,
        'reason': (
            None if can_retrain
            else (
                f'Need at least 2 classes with {min_images_per_class}+ images that are '
                f'both is_verified=True and training_ready=True.'
            )
        ),
    }


def build_temp_dataset(class_map: dict, tmp_dir: str, val_split: float):
    """
    Copy cached images into a Keras-ready split structure:

        tmp_dir/train/<class>/<image>
        tmp_dir/val/<class>/<image>

    Returns (train_dir, val_dir, dataset_info).
    """
    train_dir = os.path.join(tmp_dir, 'train')
    val_dir   = os.path.join(tmp_dir, 'val')
    dataset_info = {}

    for label, paths in class_map.items():
        shuffled = paths[:]
        random.shuffle(shuffled)

        # Clamp so train always gets at least 1 image (handles tiny test datasets)
        n_val   = min(max(1, int(len(shuffled) * val_split)), len(shuffled) - 1)
        n_train = len(shuffled) - n_val

        for split_name, split_paths in [('train', shuffled[:n_train]), ('val', shuffled[n_train:])]:
            dest_dir = os.path.join(tmp_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            for src in split_paths:
                base, ext = os.path.splitext(os.path.basename(src))
                shutil.copy2(src, os.path.join(dest_dir, f"{base}_{id(src)}{ext}"))

        dataset_info[label] = {
            'total': len(shuffled),
            'train': n_train,
            'val':   len(shuffled) - n_train,
        }

    return train_dir, val_dir, dataset_info
