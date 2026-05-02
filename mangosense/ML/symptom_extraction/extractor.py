"""
Symptom feature extraction for MangoSense training-ready images.

Ports the logic from extract-symptoms.py into a Django-runnable background
thread that reads verified+training_ready images from the database, downloads
them via the existing retrain cache mechanism, runs color/texture/lesion
feature extraction, and writes a CSV used by the Hybrid CNN during retraining.
"""

import csv
import datetime
import os
import threading
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from django.conf import settings

from .state import _set, get_extraction_status, try_start

HUE_N_BINS    = 18
GRAY_HIST_BINS = 32
IMAGE_SIZE     = (224, 224)


# ── Feature helpers (ported from extract-symptoms.py) ─────────────────────────

def _safe_entropy(values):
    values = np.asarray(values, dtype=np.float32)
    total  = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / total
    return float(-np.sum(probs * np.log2(probs + 1e-8)))


def _extract_color(rgb: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 179.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    hue_hist, _ = np.histogram(h.ravel(), bins=HUE_N_BINS, range=(0.0, 1.0), density=False)

    features = {
        'color_h_mean':            float(np.mean(h)),
        'color_h_std':             float(np.std(h)),
        'color_s_mean':            float(np.mean(s)),
        'color_s_std':             float(np.std(s)),
        'color_v_mean':            float(np.mean(v)),
        'color_v_std':             float(np.std(v)),
        'color_r_mean':            float(np.mean(rgb[:, :, 0])) / 255.0,
        'color_g_mean':            float(np.mean(rgb[:, :, 1])) / 255.0,
        'color_b_mean':            float(np.mean(rgb[:, :, 2])) / 255.0,
        'color_dominant_hue_bin':  int(np.argmax(hue_hist)) if hue_hist.size else -1,
        'color_hue_entropy':       _safe_entropy(hue_hist),
    }
    for i, val in enumerate(hue_hist):
        features[f'hue_bin_{i:02d}'] = float(val)
    return features


def _extract_texture(gray: np.ndarray) -> dict:
    gray_u8  = gray.astype(np.uint8)
    gray_f   = gray_u8.astype(np.float32) / 255.0

    lap_var  = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    sx       = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    sy       = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    canny    = cv2.Canny(gray_u8, 50, 150)
    edge_den = float(np.sum(canny > 0)) / float(canny.size)

    gh, _ = np.histogram(gray_f.ravel(), bins=GRAY_HIST_BINS, range=(0.0, 1.0), density=False)

    corr = 0.0
    if gray_f.size > 1:
        try:
            corr = float(np.corrcoef(gray_f.ravel(), np.roll(gray_f.ravel(), 1))[0, 1])
        except Exception:
            pass

    return {
        'texture_contrast':      float(np.std(gray_f)),
        'texture_energy':        float(np.mean(gray_f ** 2)),
        'texture_homogeneity':   float(1.0 / (1.0 + np.var(gray_f))),
        'texture_correlation':   corr,
        'texture_dissimilarity': float(np.mean(np.abs(np.diff(gray_f, axis=1)))) if gray_f.shape[1] > 1 else 0.0,
        'texture_laplacian_var': lap_var,
        'texture_edge_density':  edge_den,
        'texture_gradient_mean': float(np.mean(grad_mag)),
        'texture_gradient_std':  float(np.std(grad_mag)),
        'texture_gray_entropy':  _safe_entropy(gh),
    }


def _extract_lesion(rgb: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_ch = hsv[:, :, 0].astype(np.float32) / 179.0
    s_ch = hsv[:, :, 1].astype(np.float32) / 255.0
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    s_u8 = (s_ch * 255).astype(np.uint8)
    _, otsu = cv2.threshold(s_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s_thresh = float(np.mean(s_ch[otsu > 0])) if np.any(otsu > 0) else 0.3

    dark_t       = float(np.quantile(v_ch, 0.35))
    reddish      = (h_ch < 0.10) | (h_ch > 0.92)
    lesion_mask  = (
        ((s_ch > max(0.22, s_thresh)) & (v_ch < 0.90))
        | (reddish & (s_ch > 0.18) & (v_ch < 0.92))
        | (v_ch < dark_t)
    )

    kernel   = np.ones((3, 3), dtype=np.uint8)
    mask_u8  = lesion_mask.astype(np.uint8)
    mask_u8  = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel)
    mask_u8  = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    lesion_mask = mask_u8.astype(bool)

    total_px     = rgb.shape[0] * rgb.shape[1]
    lesion_ratio = float(np.sum(lesion_mask)) / float(total_px)

    count = avg = mx = std = 0.0
    if np.any(lesion_mask):
        n_labels, labels = cv2.connectedComponents(lesion_mask.astype(np.uint8))
        areas = [int(np.sum(labels == i)) for i in range(1, n_labels)]
        if areas:
            count = len(areas)
            avg   = float(np.mean(areas))
            mx    = float(np.max(areas))
            std   = float(np.std(areas)) if len(areas) > 1 else 0.0

    return {
        'lesion_ratio':    lesion_ratio,
        'lesion_count':    count,
        'lesion_avg_area': avg,
        'lesion_max_area': mx,
        'lesion_area_std': std,
    }


def _process_image(img_path: str):
    """Return (features_dict, error_str). features_dict is None on failure."""
    try:
        with Image.open(img_path) as pil:
            rgb_img  = pil.convert('RGB').resize(IMAGE_SIZE)
            rgb_arr  = np.array(rgb_img)
            gray_arr = np.array(rgb_img.convert('L'))

        features = {}
        features.update(_extract_color(rgb_arr))
        features.update(_extract_texture(gray_arr))
        features.update(_extract_lesion(rgb_arr))
        return features, None
    except Exception as exc:
        return None, str(exc)


# ── Background extraction worker ───────────────────────────────────────────────

def _run_extraction(model_type: str, output_csv: str) -> None:
    from ...models import MangoImage
    from ..retraining.cache import download_image_to_cache

    try:
        if not HAS_CV2:
            _set(
                is_running=False, phase='error', progress=0,
                message='opencv-python is not installed on this server.',
                error='cv2 not available — install opencv-python and restart the server.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='scanning', progress=5, message='Querying training-ready images from database…')

        qs = (
            MangoImage.objects
            .filter(is_verified=True, training_ready=True, disease_type=model_type)
            .exclude(disease_classification='')
            .exclude(disease_classification__isnull=True)
        )
        total = qs.count()

        if total == 0:
            _set(
                is_running=False, phase='error', progress=0,
                message='No verified training-ready images found.',
                error=f'No is_verified=True, training_ready=True images for model_type="{model_type}".',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='scanning', progress=10, message=f'Found {total} images. Downloading to cache…')

        image_records = []
        ok = fail = 0
        for img in qs:
            local_path = download_image_to_cache(img, model_type)
            if local_path and os.path.isfile(local_path):
                image_records.append({
                    'label': (img.disease_classification or '').strip(),
                    'path':  local_path,
                })
                ok += 1
            else:
                fail += 1
            pct = 10 + int(((ok + fail) / total) * 20)
            _set(progress=pct, message=f'Caching… {ok + fail}/{total} (ok: {ok}, failed: {fail})')

        if not image_records:
            _set(
                is_running=False, phase='error',
                message='All image downloads failed.',
                error='Could not download any images to the retrain cache.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='extracting', progress=30,
             message=f'Extracting features from {len(image_records)} images…')

        rows    = []
        skipped = 0
        n       = len(image_records)

        for idx, rec in enumerate(image_records):
            features, err = _process_image(rec['path'])
            if features is None:
                skipped += 1
                continue

            row = {
                'dataset':       model_type,
                'split':         '',
                'class':         rec['label'],
                'filename':      os.path.basename(rec['path']),
                'filepath':      rec['path'],
                'relative_path': rec['path'],
            }
            row.update(features)
            rows.append(row)

            if idx % max(1, n // 20) == 0:
                pct = 30 + int(((idx + 1) / n) * 60)
                _set(progress=pct,
                     message=f'Extracting… {idx + 1}/{n} (skipped: {skipped})')

        if not rows:
            _set(
                is_running=False, phase='error',
                message='Feature extraction produced no results.',
                error='No features could be extracted from the downloaded images.',
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

        _set(phase='saving', progress=92, message=f'Saving {len(rows)} rows to CSV…')

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        _set(
            is_running=False, phase='done', progress=100,
            message=f'Done. {len(rows)} features saved ({skipped} images skipped).',
            output_csv=output_csv,
            rows_extracted=len(rows),
            finished_at=datetime.datetime.now().isoformat(),
        )

    except Exception as exc:
        _set(
            is_running=False, phase='error',
            message=f'Extraction failed: {exc}',
            error=str(exc),
            finished_at=datetime.datetime.now().isoformat(),
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def start_extraction(model_type: str) -> bool:
    """
    Launch extraction in a background daemon thread.
    Returns False if an extraction job is already running.
    """
    started = try_start(
        phase='starting', progress=0,
        message='Starting symptom feature extraction…',
        started_at=datetime.datetime.now().isoformat(),
        finished_at=None, output_csv=None, rows_extracted=None, error=None,
    )
    if not started:
        return False

    output_dir = os.path.join(settings.MEDIA_ROOT, 'symptom_data')
    output_csv = os.path.join(output_dir, f'{model_type}_symptom_features.csv')

    threading.Thread(
        target=_run_extraction,
        args=(model_type, output_csv),
        daemon=True,
        name=f'mangosense-symptom-extract-{model_type}',
    ).start()
    return True


def check_symptoms_ready(model_type: str) -> dict:
    """Return whether the symptom CSV for model_type exists and how many rows it has."""
    output_dir = os.path.join(settings.MEDIA_ROOT, 'symptom_data')
    csv_path   = os.path.join(output_dir, f'{model_type}_symptom_features.csv')

    if not os.path.isfile(csv_path):
        return {'ready': False, 'csv_path': None, 'rows': None}

    rows = None
    try:
        with open(csv_path, 'r', encoding='utf-8') as fh:
            rows = max(0, sum(1 for _ in fh) - 1)
    except Exception:
        pass

    return {'ready': True, 'csv_path': csv_path, 'rows': rows}
