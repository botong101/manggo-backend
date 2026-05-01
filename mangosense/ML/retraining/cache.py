import os
import re
import shutil

from django.conf import settings

RETRAIN_CACHE_DIR = os.path.join(settings.MEDIA_ROOT, 'retrain_cache')


def _safe_label(label: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', label).strip('_') or 'unknown'


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    return re.sub(r'[^A-Za-z0-9._\-]+', '_', base) or 'image.jpg'


def _cache_path_for(model_type: str, label: str, img_id: int, image_name: str) -> str:
    return os.path.join(
        RETRAIN_CACHE_DIR,
        model_type,
        _safe_label(label),
        f"{img_id}_{_safe_filename(image_name)}",
    )


def download_image_to_cache(img, model_type: str) -> str | None:
    """
    Stream img from its storage backend (S3 or local) into the persistent cache.
    Skips download when a cached file with matching byte size already exists.
    Returns absolute local path, or None on failure.
    """
    label = (img.disease_classification or '').strip()
    if not label:
        return None

    storage = img.image.storage
    remote_name = img.image.name
    if not remote_name:
        return None

    dest = _cache_path_for(model_type, label, img.id, remote_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        remote_size = storage.size(remote_name)
    except Exception:
        remote_size = None

    if os.path.isfile(dest) and remote_size is not None:
        try:
            if os.path.getsize(dest) == remote_size:
                return dest
        except OSError:
            pass

    try:
        with storage.open(remote_name, 'rb') as src, open(dest, 'wb') as out:
            shutil.copyfileobj(src, out)
    except Exception:
        if os.path.isfile(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return None

    return dest
