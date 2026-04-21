from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..models import MangoImage
from .utils import create_api_response


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_analysis_history(request):
    """Return the authenticated user's own analysis history."""
    queryset = (
        MangoImage.objects
        .filter(user=request.user)
        .order_by('-uploaded_at')
    )

    analyses = []
    for image in queryset[:50]:
        predicted_class = image.predicted_class or 'Unknown'
        normalized = predicted_class.strip().lower()
        is_healthy = 'healthy' in normalized

        analyses.append({
            'id': image.id,
            'date': image.uploaded_at.isoformat() if image.uploaded_at else None,
            'filename': image.original_filename or 'Unknown file',
            'disease': predicted_class,
            'result': 'Healthy' if is_healthy else 'Diseased',
            'details': image.disease_classification or predicted_class,
            'confidence': image.confidence_score,
            'is_healthy': is_healthy,
            'is_verified': image.is_verified,
            'confirmed_correct': image.user_confirmed_correct,
            'location_address': image.location_address or '',
            'disease_type': image.disease_type or '',
            'image_url': request.build_absolute_uri(image.image.url) if image.image else None,
        })

    healthy_count = sum(1 for item in analyses if item['is_healthy'])
    diseased_count = len(analyses) - healthy_count

    return Response(
        create_api_response(
            success=True,
            message=f'Found {len(analyses)} analysis record(s) for the current user',
            data={
                'analyses': analyses,
                'summary': {
                    'total': len(analyses),
                    'healthy': healthy_count,
                    'diseased': diseased_count,
                },
            },
        ),
        status=200,
    )