import os
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from ..models import ModelConfig

MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')

def get_available_models(detection_type: str) -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    return sorted(files)


@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def get_model_settings(request):

    all_files = get_available_models('all')

    leaf_files  = [f for f in all_files if 'leaf' in f.lower() or 'leave' in f.lower()]
    fruit_files = [f for f in all_files if 'fruit' in f.lower()]
    other_files = [f for f in all_files if f not in leaf_files and f not in fruit_files]

    leaf_config,  _ = ModelConfig.objects.get_or_create(
        detection_type='leaf',
        defaults={'model_filename': 'leaves-mobilenetv2.keras'}
    )
    fruit_config, _ = ModelConfig.objects.get_or_create(
        detection_type='fruit',
        defaults={'model_filename': 'fruit-mobilenetv2.keras'}
    )

    return JsonResponse({
        'success': True,
        'data': {
            'available_models': {
                'all':   all_files,
                'leaf':  leaf_files  + other_files,
                'fruit': fruit_files + other_files,
            },
            'active_models': {
                'leaf':  leaf_config.model_filename,
                'fruit': fruit_config.model_filename,
            },
            'models_dir': MODELS_DIR,
        }
    })
    
@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def update_model_settings(request):
    """
    Save a new active model selection.
    Expects JSON body: { "leaf_model": "filename.keras", "fruit_model": "filename.keras" }
    Both keys are optional — send only what you want to change.
    """
    data = request.data
    updated = {}
    errors  = []

    all_files = get_available_models('all')

    for slot, key in [('leaf', 'leaf_model'), ('fruit', 'fruit_model')]:
        filename = data.get(key)
        if filename is None:
            continue  # not changing this slot
        if filename not in all_files:
            errors.append(f"File '{filename}' not found in models directory.")
            continue
        config, _ = ModelConfig.objects.get_or_create(detection_type=slot)
        config.model_filename = filename
        config.updated_by = request.user
        config.save()
        updated[slot] = filename

    if errors:
        return JsonResponse({'success': False, 'errors': errors}, status=400)

    return JsonResponse({
        'success': True,
        'message': 'Model settings updated successfully.',
        'data': {'updated': updated}
    })