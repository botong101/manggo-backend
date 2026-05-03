"""
Image preprocessing pipeline for MangoSense retraining.

Ported from preprocess-dual-branch.py:
  1. BGR → RGB
  2. Resize to IMG_SIZE using INTER_AREA (better quality for downscaling)
  3. Light Gaussian blur (3×3) — reduces sensor/camera noise while preserving
     disease features (color, texture, lesion structure)
  4. Save as uint8 PNG (normalization is handled by the model's Rescaling layer)

Preprocessed images are stored in retrain_preprocessed/{model_type}/
and are automatically preferred over raw cache files when retraining.
"""
import datetime
import os
import threading

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .state import _set, try_start

IMG_SIZE = (224, 224)


def _dirs():
    from django.conf import settings
    cache_dir  = os.path.join(settings.MEDIA_ROOT, 'retrain_cache')
    preproc_dir = os.path.join(settings.MEDIA_ROOT, 'retrain_preprocessed')
    return cache_dir, preproc_dir


def preprocessed_path_for(raw_cache_path: str) -> str:
    """Return the preprocessed counterpart path for a raw cache file."""
    cache_dir, preproc_dir = _dirs()
    rel  = os.path.relpath(raw_cache_path, cache_dir)
    base = os.path.splitext(rel)[0]
    return os.path.join(preproc_dir, base + '.png')


def _preprocess_one(src_path: str, dst_path: str) -> bool:
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        return False
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    return cv2.imwrite(dst_path, cv2.cvtColor(img_blurred, cv2.COLOR_RGB2BGR))


def check_preprocessing_ready(model_type: str) -> dict:
    """Return whether preprocessed images exist for model_type."""
    _, preproc_dir = _dirs()
    dir_path = os.path.join(preproc_dir, model_type)
    if not os.path.isdir(dir_path):
        return {'ready': False, 'processed': 0, 'classes': 0}

    total   = 0
    classes = 0
    try:
        for cls_entry in os.scandir(dir_path):
            if cls_entry.is_dir():
                count = sum(1 for f in os.scandir(cls_entry.path) if f.is_file())
                if count > 0:
                    classes += 1
                    total   += count
    except OSError:
        pass

    return {'ready': total > 0, 'processed': total, 'classes': classes}


def _run_preprocessing(model_type: str) -> None:
    from ...models import MangoImage
    from ..retraining.cache import download_image_to_cache

    try:
        if not HAS_CV2:
            _set(
                is_running=False, phase='error',
                message='opencv-python is not installed on this server.',
                error='cv2 not available — install opencv-python and restart.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='downloading', progress=5,
             message='Querying training-ready images from database…')

        qs = (
            MangoImage.objects
            .filter(is_verified=True, training_ready=True, disease_type=model_type)
            .exclude(disease_classification='')
            .exclude(disease_classification__isnull=True)
        )
        total = qs.count()
        if total == 0:
            _set(
                is_running=False, phase='error',
                message='No verified training-ready images found.',
                error=f'No images for model_type="{model_type}" with is_verified=True and training_ready=True.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(progress=8, message=f'Found {total} images. Downloading to cache…')

        raw_paths = []
        fail_dl   = 0
        for img in qs:
            local_path = download_image_to_cache(img, model_type)
            if local_path and os.path.isfile(local_path):
                raw_paths.append(local_path)
            else:
                fail_dl += 1
            done = len(raw_paths) + fail_dl
            if done % max(1, total // 10) == 0 or done == total:
                _set(progress=8 + int(done / total * 20),
                     message=f'Caching… {done}/{total} (ok: {len(raw_paths)}, failed: {fail_dl})')

        if not raw_paths:
            _set(
                is_running=False, phase='error',
                message='All image downloads failed.',
                error='No images could be downloaded to the retrain cache.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='processing', progress=30,
             message=f'Preprocessing {len(raw_paths)} images (resize → denoise)…')

        processed = 0
        failed    = 0
        n         = len(raw_paths)

        for i, raw_path in enumerate(raw_paths):
            dst = preprocessed_path_for(raw_path)
            if _preprocess_one(raw_path, dst):
                processed += 1
            else:
                failed += 1

            if i % max(1, n // 20) == 0 or i == n - 1:
                pct = 30 + int((i + 1) / n * 65)
                _set(progress=pct,
                     message=f'Preprocessing… {i + 1}/{n} — done: {processed}, failed: {failed}')

        _set(
            is_running=False, phase='done', progress=100,
            processed=processed, failed=failed,
            finished_at=datetime.datetime.now().isoformat(),
            message=f'Done. {processed} images preprocessed ({failed} skipped).',
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set(
            is_running=False, phase='error',
            error=str(exc),
            message=f'Preprocessing failed: {exc}',
            finished_at=datetime.datetime.now().isoformat(),
        )


def start_preprocessing(model_type: str) -> bool:
    """Launch preprocessing in a background daemon thread. Returns False if already running."""
    started = try_start(
        model_type=model_type, phase='starting', progress=0,
        message='Starting preprocessing job…',
        started_at=datetime.datetime.now().isoformat(),
        finished_at=None, processed=None, failed=None, error=None,
    )
    if not started:
        return False

    threading.Thread(
        target=_run_preprocessing,
        args=(model_type,),
        daemon=True,
        name=f'mangosense-preprocess-{model_type}',
    ).start()
    return True
