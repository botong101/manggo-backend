from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from mangosense.ml_constants import IMG_SIZE


def preprocess_image(image_file, target_size=None):
    try:
        img = Image.open(image_file)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
        original_size = img.size
        size = target_size if target_size else IMG_SIZE
        img = img.resize(size)
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, original_size
    except Exception as exc:
        raise exc


def get_hybrid_specs(model):
    img_hw = None
    num_features = None
    for inp in model.inputs:
        s = inp.shape
        ndim = len(s)
        if ndim == 4 and int(s[-1]) == 3:
            img_hw = (int(s[2]), int(s[1]))
        elif ndim == 2:
            num_features = int(s[1])
    return img_hw, num_features


def run_hybrid_model(model, image_file, img_size, num_features, symptom_vector=None):
    img_array, _ = preprocess_image(image_file, target_size=img_size)
    sym = symptom_vector if symptom_vector is not None \
          else np.zeros((1, num_features), dtype=np.float32)
    pred = model.predict([img_array, sym])
    return np.array(pred).flatten(), img_size
