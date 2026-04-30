import os
import json
import datetime
from pathlib import Path
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
    """Start a retraining job in the background.

    Body:
        { "model_type": "leaf" | "fruit",
          "classifier_type": "cnn" | "symptoms"   (optional, default "cnn")
        }
    Returns 409 if a job is already running.
    """
    model_type      = request.data.get('model_type', 'leaf')
    classifier_type = request.data.get('classifier_type', 'cnn')

    if model_type not in ('leaf', 'fruit'):
        return JsonResponse(
            {'success': False, 'message': "model_type must be 'leaf' or 'fruit'"},
            status=400,
        )
    if classifier_type not in ('cnn', 'symptoms'):
        return JsonResponse(
            {'success': False, 'message': "classifier_type must be 'cnn' or 'symptoms'"},
            status=400,
        )

    # ── CNN branch (unchanged behaviour) ─────────────────────────────────────
    if classifier_type == 'cnn':
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

        started = start_retraining(model_type, base_model_path, output_path, classifier_type='cnn')
        if not started:
            return JsonResponse(
                {'success': False, 'message': 'A retraining job is already running. Wait for it to finish.'},
                status=409,
            )
        return JsonResponse({
            'success': True,
            'message': f'CNN retraining started for the {model_type} model.',
            'data': {
                'classifier_type': 'cnn',
                'model_type':      model_type,
                'base_model':      os.path.basename(base_model_path),
                'output_filename': output_filename,
            },
        })

    # ── Symptoms (NB) branch ─────────────────────────────────────────────────
    from ..ML.train_naive_classifier import get_data_summary_naive
    naive_summary = get_data_summary_naive(model_type)
    if not naive_summary['can_train']:
        return JsonResponse(
            {
                'success': False,
                'message': (
                    f"Need >=2 disease classes with "
                    f"{naive_summary['min_samples_per_class']}+ verified symptom records each."
                ),
                'data': naive_summary,
            },
            status=422,
        )

    started = start_retraining(model_type, classifier_type='symptoms')
    if not started:
        return JsonResponse(
            {'success': False, 'message': 'A retraining job is already running. Wait for it to finish.'},
            status=409,
        )
    return JsonResponse({
        'success': True,
        'message': f'NB symptom-classifier retraining started for {model_type}.',
        'data': {
            'classifier_type':  'symptoms',
            'model_type':       model_type,
            'eligible_classes': naive_summary['eligible_classes'],
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


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def retrain_history(request):
    """Return past training runs from the JSONL history file.

    Query params:
        classifier_type = 'cnn' | 'symptoms' | 'all'  (default: 'all')
        model_type      = 'leaf' | 'fruit' | 'all'    (default: 'all')
        limit           = int, default 50, capped at 500
    """
    requested_classifier_type = request.query_params.get('classifier_type', 'all')
    requested_model_type      = request.query_params.get('model_type', 'all')
    try:
        requested_limit = min(int(request.query_params.get('limit', 50)), 500)
    except (TypeError, ValueError):
        requested_limit = 50

    history_file_path = Path(settings.BASE_DIR) / 'models' / 'training_history.jsonl'
    if not history_file_path.exists():
        return JsonResponse({'success': True, 'data': {'count': 0, 'runs': []}})

    parsed_history_records: list[dict] = []
    with history_file_path.open('r', encoding='utf-8') as history_file_handle:
        for raw_history_line in history_file_handle:
            stripped_history_line = raw_history_line.strip()
            if not stripped_history_line:
                continue
            try:
                parsed_history_records.append(json.loads(stripped_history_line))
            except json.JSONDecodeError:
                continue

    def matches_requested_filter(history_record: dict) -> bool:
        if requested_classifier_type != 'all':
            if history_record.get('classifier_type') != requested_classifier_type:
                return False
        if requested_model_type != 'all':
            record_model = history_record.get('plant_part') or history_record.get('model_type')
            if record_model != requested_model_type:
                return False
        return True

    filtered_history_records = [
        history_record for history_record in parsed_history_records
        if matches_requested_filter(history_record)
    ]
    filtered_history_records.reverse()

    return JsonResponse({
        'success': True,
        'data': {
            'count': len(filtered_history_records),
            'runs':  filtered_history_records[:requested_limit],
        },
    })
