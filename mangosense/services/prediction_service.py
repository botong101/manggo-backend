from __future__ import annotations

import gc
import json
import os
import time
import traceback

import numpy as np

from mangosense.ml_constants import (
    DISEASE_CONFIDENCE_THRESHOLD,
    FRUIT_CLASS_NAMES,
    IMG_SIZE,
    LEAF_CLASS_NAMES,
)
from mangosense.repositories.model_config_repository import get_active_model_path
from mangosense.repositories.prediction_repository import (
    create_mango_image,
    create_notification,
    create_prediction_log,
    rename_s3_image,
)
from mangosense.repositories.treatment_repository import get_treatment_for_disease, get_information_for_disease
from mangosense.services.gate_validation_service import GateResult
from mangosense.services.image_preprocessing_service import get_hybrid_specs, run_hybrid_model
from mangosense.services.model_loader_service import load_model


def run_prediction_pipeline(
    tf,
    image_file,
    img_array,
    original_size: tuple,
    detection_type: str,
    gate_result: GateResult,
    preview_only: bool,
    user,
    user_agent: str,
    location_data: dict,
    user_feedback: str,
    is_detection_correct: bool,
    selected_symptoms: list,
    primary_symptoms: list,
    alternative_symptoms: list,
    detected_disease: str,
    top_diseases: list,
    symptoms_data: dict,
    start_time: float,
) -> dict:
    from mangosense.views.utils import get_prediction_summary, log_prediction_activity

    if detection_type == 'fruit':
        model_path = get_active_model_path('fruit')
        model_used = 'fruit'
        model_class_names = FRUIT_CLASS_NAMES
    else:
        model_path = get_active_model_path('leaf')
        model_used = 'leaf'
        model_class_names = LEAF_CLASS_NAMES

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    model = load_model(model_path, tf)

    is_hybrid = len(model.inputs) > 1
    if is_hybrid:
        img_size, num_features = get_hybrid_specs(model)
        if img_size is None:
            img_size = (240, 240)
        if num_features is None:
            num_features = int(model.inputs[1].shape[1])

        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        proto_path = os.path.join(os.path.dirname(model_path), f"{model_stem}_prototypes.json")
        prototypes = {}
        if os.path.exists(proto_path):
            with open(proto_path) as proto_file:
                prototypes = json.load(proto_file)
            model_class_names = list(prototypes.keys())

        image_file.seek(0)
        prediction_p1, processed_size = run_hybrid_model(model, image_file, img_size, num_features)

        inference_mode = "zero_symptoms"
        symptom_vector = None
        if prototypes:
            if detected_disease and detected_disease in prototypes:
                target_class = detected_disease
                inference_mode = "user_confirmed_prototype"
            else:
                target_class = model_class_names[int(np.argmax(prediction_p1))]
                inference_mode = "self_confirmed_prototype"
            proto_values = prototypes.get(target_class)
            if proto_values is not None:
                symptom_vector = np.array([proto_values], dtype=np.float32)

        if symptom_vector is not None:
            image_file.seek(0)
            prediction, processed_size = run_hybrid_model(
                model, image_file, img_size, num_features, symptom_vector
            )
        else:
            prediction = prediction_p1
    else:
        prediction = model.predict(img_array)
        prediction = np.array(prediction).flatten()
        processed_size = IMG_SIZE
        inference_mode = "standard"

    gc.collect()

    prediction_summary = get_prediction_summary(prediction, model_class_names)
    gate_model_path = get_active_model_path(detection_type, is_gate=True)

    if prediction_summary['primary_prediction']['confidence'] < DISEASE_CONFIDENCE_THRESHOLD:
        return {
            'primary_prediction': {
                'disease': 'Unknown',
                'confidence': f"{prediction_summary['primary_prediction']['confidence']:.2f}%",
                'confidence_score': prediction_summary['primary_prediction']['confidence'],
                'confidence_level': 'Low',
                'treatment': "The uploaded image could not be confidently classified. Please ensure the image is of a mango leaf or fruit and try again.",
                'detection_type': model_used,
            },
            'top_3_predictions': [],
            'prediction_summary': {
                'most_likely': 'Unknown',
                'confidence_level': 'Low',
                'total_diseases_checked': len(model_class_names),
            },
            'alternative_symptoms': {
                'primary_disease': 'Unknown',
                'primary_disease_symptoms': [],
                'alternative_diseases': [],
            },
            'user_verification': {
                'selected_symptoms': [],
                'primary_symptoms': [],
                'alternative_symptoms': [],
                'detected_disease': 'Unknown',
                'is_detection_correct': False,
                'user_feedback': '',
            },
            'gate_validation': {
                'passed': True,
                'gate_prediction': gate_result.gate_prediction_label,
                'gate_confidence': gate_result.mango_confidence,
                'message': f'Image validated as mango {detection_type}',
            },
            'saved_image_id': None,
            'model_used': model_used,
            'model_path': model_path,
            'debug_info': {
                'gate_model_used': gate_model_path if os.path.exists(gate_model_path) else None,
                'model_loaded': True,
                'image_size': original_size,
                'processed_size': processed_size,
                'inference_mode': inference_mode,
            },
        }

    for pred in prediction_summary['top_3']:
        pred['treatment'] = get_information_for_disease(pred['disease'],model_used)
        pred['treatment'] = get_treatment_for_disease(pred['disease'],model_used)
        pred['detection_type'] = model_used

    primary_disease = prediction_summary['primary_prediction']['disease']
    alternative_diseases = [pred['disease'] for pred in prediction_summary['top_3'][1:3]]

    saved_image_id = None
    mango_image = None
    if not preview_only:
        try:
            processing_time = time.time() - start_time
            mango_image = create_mango_image(
                image_file=image_file,
                prediction_summary=prediction_summary,
                model_used=model_used,
                model_path=model_path,
                original_size=original_size,
                processing_time=processing_time,
                user=user,
                location_data=location_data,
                user_feedback=user_feedback,
                is_detection_correct=is_detection_correct,
                selected_symptoms=selected_symptoms,
                primary_symptoms=primary_symptoms,
                alternative_symptoms=alternative_symptoms,
                detected_disease=detected_disease,
                top_diseases=top_diseases,
                symptoms_data=symptoms_data,
            )
            rename_s3_image(mango_image)
            log_prediction_activity(user, mango_image.id, prediction_summary)
            saved_image_id = mango_image.id
            create_notification(mango_image, model_used, prediction_summary)
        except Exception as save_err:
            print(f"[S3 UPLOAD ERROR] {type(save_err).__name__}: {save_err}")
            print(traceback.format_exc())
            saved_image_id = None
            mango_image = None

    response_data = {
        'primary_prediction': {
            'disease': prediction_summary['primary_prediction']['disease'],
            'confidence': f"{prediction_summary['primary_prediction']['confidence']:.2f}%",
            'confidence_score': prediction_summary['primary_prediction']['confidence'],
            'confidence_level': prediction_summary['confidence_level'],
            'information': get_information_for_disease(prediction_summary['primary_prediction']['disease'],model_used),
            'treatment': get_treatment_for_disease(prediction_summary['primary_prediction']['disease'],model_used),
            'detection_type': model_used,
        },
        'top_3_predictions': prediction_summary['top_3'],
        'prediction_summary': {
            'most_likely': prediction_summary['primary_prediction']['disease'],
            'confidence_level': prediction_summary['confidence_level'],
            'total_diseases_checked': len(model_class_names),
        },
        'alternative_symptoms': {
            'primary_disease': primary_disease,
            'primary_disease_symptoms': [],
            'alternative_diseases': alternative_diseases,
        },
        'user_verification': {
            'selected_symptoms': selected_symptoms,
            'primary_symptoms': primary_symptoms,
            'alternative_symptoms': alternative_symptoms,
            'detected_disease': detected_disease,
            'is_detection_correct': is_detection_correct,
            'user_feedback': user_feedback,
        },
        'gate_validation': {
            'passed': True,
            'gate_prediction': gate_result.gate_prediction_label,
            'gate_confidence': gate_result.mango_confidence,
            'message': f'Image validated as mango {detection_type}',
        },
        'model_used': model_used,
        'model_path': model_path,
        'debug_info': {
            'gate_model_used': gate_model_path if os.path.exists(gate_model_path) else None,
            'model_loaded': True,
            'image_size': original_size,
            'processed_size': processed_size,
            'inference_mode': inference_mode,
        },
    }

    if not preview_only and saved_image_id:
        response_data['saved_image_id'] = saved_image_id

    create_prediction_log(
        mango_image=mango_image,
        prediction=prediction,
        model_class_names=model_class_names,
        prediction_summary=prediction_summary,
        response_data=response_data,
        response_time=time.time() - start_time,
        user_agent=user_agent,
    )

    return response_data
