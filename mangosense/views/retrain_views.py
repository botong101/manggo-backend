import os
import dataclasses
import datetime
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser

from ..ML.retrain import start_retraining, get_status, get_dataset_preview, RetrainConfig
from ..ML.symptom_extraction import get_extraction_status, start_extraction, check_symptoms_ready
from .ml_views import get_active_model_path

MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')


@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def trigger_retrain(request):
    """
    Start a retraining job in the background.

    Request body:
        { "model_type": "leaf" | "fruit" }

    Returns 409 if a job is already running.
    """
    model_type = request.data.get('model_type', 'leaf')
    model_kind = request.data.get('model_kind', 'mobilenetv2')
    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )
    if model_kind not in ('mobilenetv2', 'hybrid_cnn'):
        return JsonResponse(
            {'success': False, 'message': "model_kind must be 'mobilenetv2' or 'hybrid_cnn'"},
            status=400,
        )

    # Quick dataset check before starting
    preview = get_dataset_preview(model_type)
    if not preview['can_retrain']:
        return JsonResponse(
            {'success': False, 'message': preview['reason'], 'data': preview},
            status=422,
        )

    base_model_path = get_active_model_path(model_type)
    timestamp       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'{model_type}-retrained-{timestamp}.keras'
    output_path     = os.path.join(MODELS_DIR, output_filename)

    config = RetrainConfig(
        epochs=int(request.data.get('epochs', 10)),
        learning_rate=float(request.data.get('learning_rate', 1e-4)),
        batch_size=int(request.data.get('batch_size', 16)),
        val_split=float(request.data.get('val_split', 0.2)),
        unfreeze_top_n_layers=int(request.data.get('unfreeze_top_n_layers', 20)),
        early_stopping_patience=int(request.data.get('early_stopping_patience', 3)),
        lr_reduce_factor=float(request.data.get('lr_reduce_factor', 0.5)),
        lr_reduce_patience=int(request.data.get('lr_reduce_patience', 2)),
        min_images_per_class=int(request.data.get('min_images_per_class', 5)),
    )
    started = start_retraining(model_type, base_model_path, output_path, config, model_kind=model_kind)
    if not started:
        return JsonResponse(
            {'success': False, 'message': 'A retraining job is already running. Wait for it to finish.'},
            status=409,
        )

    return JsonResponse({
        'success': True,
        'message': f'Retraining started for the {model_type} model ({model_kind}).',
        'data': {
            'model_type':      model_type,
            'model_kind':      model_kind,
            'base_model':      os.path.basename(base_model_path),
            'output_filename': output_filename,
            'config': dataclasses.asdict(config),
        },
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def retrain_status(request):
    """
    Poll the current retraining job status.

    Returns the full status dict:
        is_running, phase, progress, message, accuracy, output_filename, error, …
    """
    status = get_status()
    return JsonResponse({'success': True, 'data': status})


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def retrain_dataset_info(request):
    """
    Preview the verified-image dataset for a given model_type
    without starting training.

    Query param:  ?model_type=leaf   (default: leaf)

    Returns per-class image counts and whether retraining is possible.
    """
    model_type = request.query_params.get('model_type', 'leaf')
    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )

    preview = get_dataset_preview(model_type)
    return JsonResponse({'success': True, 'data': preview})


# ── Symptom feature extraction (Hybrid CNN prerequisite) ──────────────────────

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def trigger_symptom_extraction(request):
    """
    Start symptom feature extraction in the background.

    Scans verified+training_ready images, extracts color/texture/lesion
    features, and saves them to a CSV that the Hybrid CNN training job reads.

    Request body:
        { "model_type": "leaf" | "fruit" }

    Returns 409 if extraction is already running.
    """
    model_type = request.data.get('model_type', 'leaf')
    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )

    started = start_extraction(model_type)
    if not started:
        return JsonResponse(
            {'success': False, 'message': 'A symptom extraction job is already running.'},
            status=409,
        )

    return JsonResponse({
        'success': True,
        'message': f'Symptom feature extraction started for the {model_type} dataset.',
        'data': {'model_type': model_type},
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def symptom_extraction_status(request):
    """
    Poll the current symptom extraction job status.

    Returns: is_running, phase, progress, message, output_csv, rows_extracted, error
    """
    status = get_extraction_status()
    return JsonResponse({'success': True, 'data': status})


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def symptoms_ready(request):
    """
    Check whether the symptom CSV for a given model_type already exists.

    Query param: ?model_type=leaf  (default: leaf)

    Returns: { ready: bool, csv_path: str|null, rows: int|null }
    """
    model_type = request.query_params.get('model_type', 'leaf')
    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )

    result = check_symptoms_ready(model_type)
    return JsonResponse({'success': True, 'data': result})
