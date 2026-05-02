from __future__ import annotations

import json
import os
import time
from mangosense.models import MangoImage, MLModel, Disease
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated

from mangosense.ml_constants import (
    FRUIT_CLASS_NAMES,
    GATE_CONFIDENCE_THRESHOLD_FRUIT,
    GATE_CONFIDENCE_THRESHOLD_LEAF,
    GATE_FRUIT_CLASS_NAMES,
    GATE_LEAF_CLASS_NAMES,
    IMG_SIZE,
    LEAF_CLASS_NAMES,
    #treatment_suggestions,
)
from mangosense.models import MangoImage, MLModel
from mangosense.repositories.model_config_repository import get_active_model_path  # re-export for callers
from mangosense.services.gate_validation_service import run_gate_validation
from mangosense.services.image_preprocessing_service import preprocess_image
from mangosense.services.model_loader_service import get_tensorflow_runtime
from mangosense.services.prediction_service import run_prediction_pipeline
from mangosense.views.utils import create_api_response, validate_image_file


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([IsAuthenticated])
def predict_image(request):
    start_time = time.time()

    if 'image' not in request.FILES:
        return JsonResponse(
            create_api_response(success=False, message='No image uploaded', errors=['Image file is required']),
            status=400,
        )

    try:
        tf, tf_runtime_error = get_tensorflow_runtime()
        if tf is None:
            return JsonResponse(
                create_api_response(success=False, message='TensorFlow runtime unavailable', errors=[tf_runtime_error]),
                status=503,
            )

        image_file = request.FILES['image']
        preview_only = request.data.get('preview_only', 'false').lower() == 'true'
        is_detection_correct = request.data.get('is_detection_correct', '').lower() == 'true'
        user_feedback = request.data.get('user_feedback', '')
        detection_type = request.data.get('detection_type', 'leaf')
        detected_disease = request.data.get('detected_disease', '')

        def _parse_json_field(key, default):
            raw = request.data.get(key)
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        selected_symptoms = _parse_json_field('selected_symptoms', [])
        primary_symptoms = _parse_json_field('primary_symptoms', [])
        alternative_symptoms = _parse_json_field('alternative_symptoms', [])
        top_diseases = _parse_json_field('top_diseases', [])
        symptoms_data = _parse_json_field('symptoms_data', {})

        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        location_accuracy_confirmed = request.data.get('location_accuracy_confirmed', 'false').lower() == 'true'
        location_source = request.data.get('location_source', '')
        location_address = request.data.get('location_address', '')

        validation_errors = validate_image_file(image_file)
        if validation_errors:
            return JsonResponse(
                create_api_response(success=False, message='Invalid image file', errors=validation_errors),
                status=400,
            )

        try:
            img_array, original_size = preprocess_image(image_file)
        except Exception as preprocessing_error:
            return JsonResponse(
                create_api_response(success=False, message='Image preprocessing failed', errors=[str(preprocessing_error)]),
                status=500,
            )

        gate_model_path = get_active_model_path(detection_type, is_gate=True)
        gate_result = run_gate_validation(tf, img_array, gate_model_path, detection_type)

        if not gate_result.passed:
            part = detection_type.capitalize()
            if gate_result.gate_prediction_label and gate_result.gate_prediction_label.lower() != 'mango':
                rejection_detail = f'The image appears to be a {gate_result.gate_prediction_label} {detection_type}, not a Mango {detection_type}.'
            else:
                rejection_detail = f'The image does not appear to be a clear Mango {detection_type}. Mango confidence: {gate_result.mango_confidence:.1f}%'

            return JsonResponse(
                create_api_response(
                    success=True,
                    data={
                        'primary_prediction': {
                            'disease': f'Not a Mango {part}',
                            'confidence': f"{gate_result.mango_confidence:.2f}%",
                            'confidence_score': gate_result.mango_confidence or 0,
                            'confidence_level': 'Low',
                            'treatment': (
                                f"{rejection_detail} "
                                f"Please upload a clear image of a mango {detection_type} and try again."
                            ),
                            'detection_type': detection_type,
                        },
                        'top_3_predictions': [],
                        'prediction_summary': {
                            'most_likely': f'Not a Mango {part}',
                            'confidence_level': 'Low',
                            'total_diseases_checked': 0,
                        },
                        'alternative_symptoms': {
                            'primary_disease': f'Not a Mango {part}',
                            'primary_disease_symptoms': [],
                            'alternative_diseases': [],
                        },
                        'user_verification': {
                            'selected_symptoms': [],
                            'primary_symptoms': [],
                            'alternative_symptoms': [],
                            'detected_disease': f'Not a Mango {part}',
                            'is_detection_correct': False,
                            'user_feedback': '',
                        },
                        'gate_validation': {
                            'passed': False,
                            'gate_prediction': gate_result.gate_prediction_label,
                            'gate_predicted_confidence': gate_result.gate_predicted_confidence,
                            'mango_confidence': gate_result.mango_confidence,
                            'threshold': gate_result.threshold,
                            'message': rejection_detail,
                        },
                        'saved_image_id': None,
                        'model_used': detection_type,
                        'debug_info': {
                            'gate_model_used': True,
                            'gate_model_path': gate_model_path,
                            'processing_time': time.time() - start_time,
                            'image_size': original_size,
                            'inference_mode': 'gate_rejected',
                        },
                    },
                    message=rejection_detail,
                )
            )

        if latitude and longitude:
            try:
                location_data = {
                    'latitude': float(latitude),
                    'longitude': float(longitude),
                    'location_consent_given': True,
                    'location_accuracy_confirmed': location_accuracy_confirmed,
                    'location_source': location_source,
                    'location_address': location_address,
                }
            except (ValueError, TypeError):
                location_data = {
                    'location_consent_given': False,
                    'location_accuracy_confirmed': False,
                }
        else:
            location_data = {
                'location_consent_given': False,
                'location_accuracy_confirmed': False,
            }

        try:
            response_data = run_prediction_pipeline(
                tf=tf,
                image_file=image_file,
                img_array=img_array,
                original_size=original_size,
                detection_type=detection_type,
                gate_result=gate_result,
                preview_only=preview_only,
                user=request.user,
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                location_data=location_data,
                user_feedback=user_feedback,
                is_detection_correct=is_detection_correct,
                selected_symptoms=selected_symptoms,
                primary_symptoms=primary_symptoms,
                alternative_symptoms=alternative_symptoms,
                detected_disease=detected_disease,
                top_diseases=top_diseases,
                symptoms_data=symptoms_data,
                start_time=start_time,
            )
        except FileNotFoundError as model_err:
            return JsonResponse(
                create_api_response(success=False, message=str(model_err), errors=[str(model_err)]),
                status=500,
            )

        return JsonResponse(
            create_api_response(success=True, data=response_data, message='Image processed successfully')
        )

    except Exception as exc:
        return JsonResponse(
            create_api_response(success=False, message='Prediction failed', errors=[str(exc)]),
            status=500,
        )


@api_view(['GET'])
def test_model_status(request):
    try:
        active_model = MLModel.objects.filter(is_active=True).first()

        leaf_model_path = get_active_model_path('leaf')
        fruit_model_path = get_active_model_path('fruit')
        gate_leaf_model_path = get_active_model_path('leaf', is_gate=True)
        gate_fruit_model_path = get_active_model_path('fruit', is_gate=True)

        model_status = {
            'model_loaded': active_model is not None,
            'leaf_model_path': leaf_model_path,
            'fruit_model_path': fruit_model_path,
            'gate_leaf_model_path': gate_leaf_model_path,
            'gate_fruit_model_path': gate_fruit_model_path,
            'leaf_model_exists': os.path.exists(leaf_model_path),
            'fruit_model_exists': os.path.exists(fruit_model_path),
            'gate_leaf_model_exists': os.path.exists(gate_leaf_model_path),
            'gate_fruit_model_exists': os.path.exists(gate_fruit_model_path),
            'leaf_class_names': LEAF_CLASS_NAMES,
            'fruit_class_names': FRUIT_CLASS_NAMES,
            'gate_leaf_class_names': GATE_LEAF_CLASS_NAMES,
            'gate_fruit_class_names': GATE_FRUIT_CLASS_NAMES,
            'leaf_classes_count': len(LEAF_CLASS_NAMES),
            'fruit_classes_count': len(FRUIT_CLASS_NAMES),
            'treatment_suggestions_count': Disease.objects.exclude(treatment__isnull=True).exclude(treatment__exact='').count(),
            'gate_confidence_threshold_leaf': GATE_CONFIDENCE_THRESHOLD_LEAF,
            'gate_confidence_threshold_fruit': GATE_CONFIDENCE_THRESHOLD_FRUIT,
            'active_model': {
                'name': active_model.name if active_model else None,
                'version': active_model.version if active_model else None,
                'file_path': active_model.file_path if active_model else None,
            } if active_model else None,
            'img_size': IMG_SIZE,
        }

        database_stats = {
            'total_images': MangoImage.objects.count(),
            'healthy_images': MangoImage.objects.filter(disease_classification='Healthy').count(),
            'diseased_images': MangoImage.objects.exclude(disease_classification='Healthy').count(),
            'verified_images': MangoImage.objects.filter(is_verified=True).count(),
        }

        return JsonResponse(
            create_api_response(
                success=True,
                data={
                    'model_status': model_status,
                    'available_leaf_diseases': LEAF_CLASS_NAMES,
                    'available_fruit_diseases': FRUIT_CLASS_NAMES,
                    'database_stats': database_stats,
                },
                message='Model status retrieved successfully',
            )
        )

    except Exception as exc:
        return JsonResponse(
            create_api_response(success=False, message='Failed to get model status', errors=[str(exc)]),
            status=500,
        )
