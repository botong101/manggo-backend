from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.db.models import Count, Q
from django.utils import timezone
from ..models import MangoImage, UserConfirmation
from .utils import get_client_ip, create_api_response
import json

@api_view(['POST'])
@permission_classes([AllowAny])
def save_user_confirmation(request):
    """save user feedback for ai prediction"""
    try:
        data = request.data
        
        # need these
        image_id = data.get('image_id')
        predicted_disease = data.get('predicted_disease')
        
        if image_id is None or not predicted_disease:
            missing_fields = []
            if image_id is None:
                missing_fields.append('image_id')
            if not predicted_disease:
                missing_fields.append('predicted_disease')
            
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Missing required fields',
                    errors=[f'Missing fields: {", ".join(missing_fields)}']
                ),
                status=400
            )
        
        # get the image
        try:
            image = MangoImage.objects.get(id=image_id)
        except MangoImage.DoesNotExist:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Image not found',
                    errors=[f'No image found with ID {image_id}']
                ),
                status=404
            )
        
        # check if already confirmed
        existing_confirmation = UserConfirmation.objects.filter(image=image).first()
        if existing_confirmation:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Confirmation already exists for this image',
                    errors=['This image has already been confirmed by user']
                ),
                status=400
            )
        
        # make confirmation record
        confirmation_data = {
            'image': image,
            'user': request.user if request.user.is_authenticated else None,
            'predicted_disease': predicted_disease,
            'user_feedback': data.get('user_feedback', ''),
            'confidence_score': data.get('confidence_score'),
        }
        
        # handle gps stuff if user said ok
        location_consent = data.get('location_consent_given', False)
        
        if location_consent:
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            location_accuracy = data.get('location_accuracy')
            location_address = data.get('location_address', '')
            
            confirmation_data.update({
                'location_consent_given': True,
                'latitude': latitude,
                'longitude': longitude,
                'location_accuracy': location_accuracy,
                'location_address': location_address,
            })
        
        confirmation = UserConfirmation.objects.create(**confirmation_data)
        
        response_data = {
            'confirmation_id': confirmation.id,
            'image_id': image.id,
            'predicted_disease': confirmation.predicted_disease,
            'location_saved': confirmation.location_consent_given
        }
        
        
        return JsonResponse(
            create_api_response(
                success=True,
                data=response_data,
                message='User confirmation saved successfully'
            )
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return JsonResponse(
            create_api_response(
                success=False,
                message='Failed to save confirmation',
                errors=[str(e)]
            ),
            status=500
        )
        
        confirmation = UserConfirmation.objects.create(**confirmation_data)
        
        return JsonResponse(
            create_api_response(
                success=True,
                data={
                    'confirmation_id': confirmation.id,
                    'image_id': image.id,
                    'predicted_disease': confirmation.predicted_disease,
                    'location_saved': confirmation.location_consent_given
                },
                message='User confirmation saved successfully'
            )
        )
        
    except Exception as e:
        return JsonResponse(
            create_api_response(
                success=False,
                message='Failed to save confirmation',
                errors=[str(e)]
            ),
            status=500
        )

@api_view(['GET'])
@permission_classes([AllowAny])  # Temporarily allow any access for debugging
def get_user_confirmations(request):
    """get confirmations list for admin dashboard"""
    try:
        
        # get filter stuff
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        filter_type = request.GET.get('filter', 'all')  # all, confirmed, rejected
        user_id = request.GET.get('user_id')
        disease = request.GET.get('disease')
        image_id = request.GET.get('image_id')  # Add this filter for admin panel
        
        # get all confirmations
        queryset = UserConfirmation.objects.select_related('image', 'user').all()
        
        # apply filters (is_correct removed so filter_type not used)
        
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        
        if disease:
            queryset = queryset.filter(predicted_disease__icontains=disease)
            
        # add image filter for admin
        if image_id:
            try:
                image_id_int = int(image_id)
                queryset = queryset.filter(image_id=image_id_int)
            except (ValueError, TypeError):
                pass

        # paging stuff
        total_count = queryset.count()
        
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        confirmations = queryset[start_index:end_index]
        
        
        # turn into json
        confirmation_data = []
        for conf in confirmations:
            
            confirmation_item = {
                'id': conf.id,
                'image_id': conf.image.id,  # Add this for admin panel compatibility
                'image': {
                    'id': conf.image.id,
                    'filename': conf.image.original_filename,
                    'image_url': conf.image.image.url if conf.image.image else None,
                    'uploaded_at': conf.image.uploaded_at.isoformat(),
                },
                'user': {
                    'id': conf.user.id if conf.user else None,
                    'username': conf.user.username if conf.user else 'Anonymous',
                    'email': conf.user.email if conf.user else '',
                    'full_name': f"{conf.user.first_name} {conf.user.last_name}".strip() if conf.user else 'Anonymous'
                },
                'predicted_disease': conf.predicted_disease,
                'user_feedback': conf.user_feedback,
                'confidence_score': conf.confidence_score,
                'location_consent_given': conf.location_consent_given,  # Add this for admin panel
                'latitude': conf.latitude,  # Add direct fields for admin panel
                'longitude': conf.longitude,
                'location_accuracy': conf.location_accuracy,
                'location_address': conf.location_address,
                'location': {
                    'consent_given': conf.location_consent_given,
                    'latitude': conf.latitude,
                    'longitude': conf.longitude,
                    'accuracy': conf.location_accuracy,
                    'address': conf.location_address
                } if conf.location_consent_given else None
            }
            
            confirmation_data.append(confirmation_item)
        
        # get stats
        stats = {
            'total_confirmations': total_count
        }
        
        return JsonResponse(
            create_api_response(
                success=True,
                data={
                    'confirmations': confirmation_data,
                    'pagination': {
                        'page': page,
                        'page_size': page_size,
                        'total_count': total_count,
                        'total_pages': (total_count + page_size - 1) // page_size,
                        'has_next': end_index < total_count,
                        'has_previous': page > 1
                    },
                    'statistics': stats
                },
                message='User confirmations retrieved successfully'
            )
        )
        
    except Exception as e:
        return JsonResponse(
            create_api_response(
                success=False,
                message='Failed to get confirmations',
                errors=[str(e)]
            ),
            status=500
        )

@api_view(['GET'])
@permission_classes([AllowAny])  # Temporarily allow any access for debugging  
def get_confirmation_statistics(request):
    """get stats about user confirmations"""
    try:
        # overall stats
        total_confirmations = UserConfirmation.objects.count()
        
        # disease stats
        disease_stats = []
        diseases = UserConfirmation.objects.values('predicted_disease').distinct()
        
        for disease_data in diseases:
            disease = disease_data['predicted_disease']
            disease_total = UserConfirmation.objects.filter(predicted_disease=disease).count()
            
            disease_stats.append({
                'disease': disease,
                'total_predictions': disease_total
            })
        
        # sort by most common
        disease_stats.sort(key=lambda x: x['total_predictions'], reverse=True)
        
        # user stats
        users_with_confirmations = UserConfirmation.objects.filter(
            user__isnull=False
        ).values('user').distinct().count()
        
        anonymous_confirmations = UserConfirmation.objects.filter(
            user__isnull=True
        ).count()
        
        # gps data stats
        confirmations_with_location = UserConfirmation.objects.filter(
            location_consent_given=True
        ).count()
        
        return JsonResponse(
            create_api_response(
                success=True,
                data={
                    'overall_statistics': {
                        'total_confirmations': total_confirmations,
                        'users_with_confirmations': users_with_confirmations,
                        'anonymous_confirmations': anonymous_confirmations,
                        'confirmations_with_location': confirmations_with_location
                    },
                    'disease_statistics': disease_stats,
                    'location_consent_rate': round(
                        (confirmations_with_location / total_confirmations * 100), 2
                    ) if total_confirmations > 0 else 0
                },
                message='Confirmation statistics retrieved successfully'
            )
        )
        
    except Exception as e:
        return JsonResponse(
            create_api_response(
                success=False,
                message='Failed to get statistics',
                errors=[str(e)]
            ),
            status=500
        )
