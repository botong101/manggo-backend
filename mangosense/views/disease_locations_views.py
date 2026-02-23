from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from ..models import MangoImage
from .utils import create_api_response


@api_view(['GET'])
@permission_classes([AllowAny])
def disease_locations_similar(request):
    """
    Return stored locations whose primary AI prediction matches the requested disease.
    Query params:
        disease  (required)  – disease name to match (case-insensitive)
    """
    disease = request.GET.get('disease', '').strip()

    if not disease:
        return Response(
            create_api_response(
                success=False,
                message='disease query parameter is required',
                errors=['Missing: disease']
            ),
            status=400,
        )

    qs = (
        MangoImage.objects
        .filter(
            predicted_class__iexact=disease,
            latitude__isnull=False,
            longitude__isnull=False,
        )
        .values(
            'id', 'predicted_class', 'latitude', 'longitude',
            'location_address', 'uploaded_at', 'confidence_score',
        )
    )

    locations = [
        {
            'id': img['id'],
            'disease': img['predicted_class'],
            'latitude': img['latitude'],
            'longitude': img['longitude'],
            'address': img['location_address'] or '',
            'uploaded_at': img['uploaded_at'].isoformat() if img['uploaded_at'] else None,
            'confidence': img['confidence_score'],
        }
        for img in qs
    ]

    return Response(
        create_api_response(
            success=True,
            message=f'Found {len(locations)} location(s) with similar disease',
            data={'locations': locations},
        ),
        status=200,
    )


@api_view(['GET'])
@permission_classes([AllowAny])
def disease_locations_all(request):
    """
    Return all stored detection locations regardless of disease type.
    """
    qs = (
        MangoImage.objects
        .filter(
            latitude__isnull=False,
            longitude__isnull=False,
        )
        .values(
            'id', 'predicted_class', 'latitude', 'longitude',
            'location_address', 'uploaded_at', 'confidence_score',
        )
    )

    locations = [
        {
            'id': img['id'],
            'disease': img['predicted_class'] or 'Unknown',
            'latitude': img['latitude'],
            'longitude': img['longitude'],
            'address': img['location_address'] or '',
            'uploaded_at': img['uploaded_at'].isoformat() if img['uploaded_at'] else None,
            'confidence': img['confidence_score'],
        }
        for img in qs
    ]

    return Response(
        create_api_response(
            success=True,
            message=f'Found {len(locations)} total detection location(s)',
            data={'locations': locations},
        ),
        status=200,
    )
