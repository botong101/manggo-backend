import os
import datetime
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser

from ..ML.retrain import start_retraining, get_status, get_dataset_preview
from .ml_views import get_active_model_path

MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')


@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def trigger_retrain(request):
    """
    Start a retraining job in the background.

    Request body:
        { 
            "model_type": "leaf" | "fruit",
            "model_variant": "standard" | "hybrid"  (optional, defaults to 'standard')
        }

    Returns 409 if a job is already running.
    """
    model_type = request.data.get('model_type', 'leaf')
    model_variant = request.data.get('model_variant', 'standard')
    
    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )
    
    if model_variant not in ('standard', 'hybrid'):
        return JsonResponse(
            {'success': False, 'message': "model_variant must be 'standard' or 'hybrid'"},
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
    output_filename = f'{model_type}-{model_variant}-retrained-{timestamp}.keras'
    output_path     = os.path.join(MODELS_DIR, output_filename)

    started = start_retraining(model_type, base_model_path, output_path, model_variant)
    if not started:
        return JsonResponse(
            {'success': False, 'message': 'A retraining job is already running. Wait for it to finish.'},
            status=409,
        )

    return JsonResponse({
        'success': True,
        'message': f'Retraining started for the {model_type} model ({model_variant} variant).',
        'data': {
            'model_type':      model_type,
            'model_variant':   model_variant,
            'base_model':      os.path.basename(base_model_path),
            'output_filename': output_filename,
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
