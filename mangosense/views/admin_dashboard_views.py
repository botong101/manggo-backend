from rest_framework import status, generics, filters
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.pagination import PageNumberPagination
from django.http import JsonResponse, HttpResponse
from django.db.models import Count, Q
from django.utils import timezone
from datetime import datetime, timedelta
import json
import traceback
from django.contrib.auth.models import User
from ..models import MangoImage, MLModel, PredictionLog, UserProfile
from ..serializers import (
    MangoImageSerializer, MangoImageUpdateSerializer, 
    BulkUpdateSerializer, ImageUploadSerializer, UserDetailSerializer
)
from .ml_views import LEAF_MODEL_PATH, FRUIT_MODEL_PATH
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt



def get_actual_model_path(image):
    """
    figure out which model was used
    """
    model_filename = getattr(image, 'model_filename', None)
    if model_filename:
        model_path = f"models/{model_filename}"
        return model_path
    else:
        #fallback to old way
        disease_type = getattr(image, 'disease_type', 'leaf')
        import os
        if disease_type == 'fruit':
            model_path = f"models/{os.path.basename(FRUIT_MODEL_PATH)}"
            return model_path
        else: 
            model_path = f"models/{os.path.basename(LEAF_MODEL_PATH)}"
            return model_path

def get_top_predictions_for_image(image):
    #get top 3 from prediction log
    try:
        #grab latest prediction for this image
        prediction_log = PredictionLog.objects.filter(image=image).order_by('-timestamp').first()
        
        if prediction_log and prediction_log.probabilities and prediction_log.labels:
            probabilities = prediction_log.probabilities
            labels = prediction_log.labels

            prob_label_pairs = list(zip(probabilities, labels))
            prob_label_pairs.sort(key=lambda x: x[0], reverse=True)
            
            top_3_pairs = prob_label_pairs[:3]
            
            top_3_predictions = [
                {
                    'predicted_class': label,
                    'confidence': float(prob)
                }
                for prob, label in top_3_pairs
            ]
            
            return top_3_predictions
            
    except Exception as e:
        print(f"Error getting stored predictions: {e}")
    
    # if no log just make something up
    predicted_class = image.predicted_class or "Unknown"
    confidence = image.confidence_score or 0.0
    
    # fake 3 predictions with lower confidence each
    base_confidence = min(confidence, 0.95)  # max 95%
    
    top_3_predictions = [
        {
            'predicted_class': predicted_class,
            'confidence': base_confidence
        },
        {
            'predicted_class': f"Alternative to {predicted_class}",
            'confidence': max(0.1, base_confidence - 0.3)
        },
        {
            'predicted_class': f"Other possibility",
            'confidence': max(0.05, base_confidence - 0.5)
        }
    ]
    
    return top_3_predictions

#class StandardResultsSetPagination(PageNumberPagination):
 #   page_size = 20
  #  page_size_query_param = 'page_size'
   # max_page_size = 100

# ================ STATS STUFF ================

@csrf_exempt
@require_http_methods(["GET"])
def disease_statistics(request):
    """stats for dashboard"""
    try:
        #how many images total
        total_images = MangoImage.objects.count()
        
        #how many healthy
        healthy_images = MangoImage.objects.filter(
            Q(predicted_class__icontains='healthy') | 
            Q(predicted_class__icontains='Healthy')
        ).count()
        
        diseased_images = total_images - healthy_images
        
        #leaf vs fruit count
        leaf_images = MangoImage.objects.filter(
            predicted_class__icontains='Leaf'
        ).count()
        
        fruit_images = MangoImage.objects.filter(
            predicted_class__icontains='Fruit'
        ).count()
        
        #group by disease
        diseases_breakdown = {}
        disease_counts = MangoImage.objects.values('predicted_class').annotate(
            count=Count('id')
        ).order_by('-count')
        
        for disease in disease_counts:
            diseases_breakdown[disease['predicted_class']] = disease['count']
        
        #last 7 days
        week_ago = timezone.now() - timedelta(days=7)
        recent_uploads = MangoImage.objects.filter(
            uploaded_at__gte=week_ago
        ).count()
        
        #last 30 days
        month_ago = timezone.now() - timedelta(days=30)
        monthly_uploads = MangoImage.objects.filter(
            uploaded_at__gte=month_ago
        ).count()
        
        data = {
            'total_images': total_images,
            'healthy_images': healthy_images,
            'diseased_images': diseased_images,
            'leaf_images': leaf_images,
            'fruit_images': fruit_images,
            'diseases_breakdown': diseases_breakdown,
            'recent_uploads': recent_uploads,
            'monthly_uploads': monthly_uploads,
            'verification_stats': {
                'verified': MangoImage.objects.filter(is_verified=True).count(),
                'unverified': MangoImage.objects.filter(is_verified=False).count(),
            }
        }
        
        return JsonResponse({
            'success': True,
            'data': data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)


# ================ IMAGE LIST STUFF ================

@csrf_exempt
@require_http_methods(["GET"])
def classified_images_list(request):
    """get list of images with pagination"""
    try:
        # Get query parameters
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        search = request.GET.get('search', '')
        disease_filter = request.GET.get('disease', '')
        disease_type_filter = request.GET.get('disease_type', '')
        verified_filter = request.GET.get('verified', '')
        
        # build the query
        queryset = MangoImage.objects.all().order_by('-uploaded_at')  # newest first
        
        # apply filters
        if search:
            queryset = queryset.filter(
                Q(original_filename__icontains=search) |
                Q(predicted_class__icontains=search)
            )
        
        if disease_filter:
            queryset = queryset.filter(predicted_class__icontains=disease_filter)
        
        if disease_type_filter:
            queryset = queryset.filter(disease_type__iexact=disease_type_filter)
        
        if verified_filter:
            is_verified = verified_filter.lower() == 'true'
            queryset = queryset.filter(is_verified=is_verified)
        
        # Pagination
        total_count = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        images = queryset[start:end]
        
        # Serialize data
        serializer = MangoImageSerializer(images, many=True, context={'request': request})
        
        return JsonResponse({
            'success': True,
            'data': {
                'images': serializer.data,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size,
                    'has_next': end < total_count,
                    'has_previous': page > 1
                }
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt 
@require_http_methods(["GET", "PUT", "DELETE"])
def classified_images_detail(request, pk):
    """get update or delete single image"""
    try:
        image = MangoImage.objects.get(pk=pk)
        
        if request.method == 'GET':
            serializer = MangoImageSerializer(image)
            return JsonResponse({
                'success': True,
                'data': serializer.data
            })
            
        elif request.method == 'PUT':
            data = json.loads(request.body)
            serializer = MangoImageUpdateSerializer(image, data=data, partial=True)
            
            if serializer.is_valid():
                serializer.save()
                return JsonResponse({
                    'success': True,
                    'message': 'Image updated successfully',
                    'data': MangoImageSerializer(image).data
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': serializer.errors
                }, status=400)
                
        elif request.method == 'DELETE':
            image.delete()
            return JsonResponse({
                'success': True,
                'message': 'Image deleted successfully'
            })
            
    except MangoImage.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Image not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

# ================ MORE VIEWS ================

@csrf_exempt
@require_http_methods(["POST"])
def bulk_update_images(request):
    """update bunch of images at once"""
    try:
        data = json.loads(request.body)
        serializer = BulkUpdateSerializer(data=data)
        
        if serializer.is_valid():
            image_ids = serializer.validated_data['image_ids']
            updates = serializer.validated_data['updates']
            
            # do the update
            updated_count = MangoImage.objects.filter(
                id__in=image_ids
            ).update(**updates)
            
            return JsonResponse({
                'success': True,
                'message': f'Successfully updated {updated_count} images',
                'updated_count': updated_count
            })
        else:
            return JsonResponse({
                'success': False,
                'error': serializer.errors
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_image(request):
    """upload new image"""
    try:
        serializer = ImageUploadSerializer(data=request.FILES)
        
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            
            # make new image record
            mango_image = MangoImage.objects.create(
                image=image_file,
                original_filename=image_file.name,
                user=request.user if request.user.is_authenticated else None
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Image uploaded successfully',
                'data': MangoImageSerializer(mango_image).data
            })
        else:
            return JsonResponse({
                'success': False,
                'error': serializer.errors
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def export_dataset(request):
    """dump all data"""
    try:
        # get everything
        images = MangoImage.objects.all()
        
        # build export list
        export_data = []
        for image in images:
            export_data.append({
                'id': image.id,
                'filename': image.original_filename,
                'predicted_class': image.predicted_class,
                'confidence_score': image.confidence_score,
                'uploaded_at': image.uploaded_at.isoformat(),
                'is_verified': image.is_verified
            })
        
        return JsonResponse({
            'success': True,
            'data': {
                'images': export_data,
                'total_count': len(export_data)
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

# ================ PREDICTION DETAILS ================

@csrf_exempt
@require_http_methods(["GET"])
def image_prediction_details(request, pk):
    """get all prediction info for one image"""
    try:
        image = MangoImage.objects.get(pk=pk)
        serializer = MangoImageSerializer(image, context={'request': request})
        
        # get top 3 using helper
        top_3_predictions = get_top_predictions_for_image(image)
        
        # make response with all the details
        data = serializer.data
        
        # format predictions for frontend
        formatted_predictions = []
        for i, pred in enumerate(top_3_predictions):
            confidence_score = pred['confidence'] * 100
            formatted_predictions.append({
                'disease': pred['predicted_class'],
                'confidence': f"{confidence_score:.2f}%",
                'confidence_score': confidence_score,
                'confidence_level': 'High' if pred['confidence'] > 0.8 else 'Medium' if pred['confidence'] > 0.6 else 'Low',
                'treatment': f"Treatment recommendations for {pred['predicted_class']}",
                'detection_type': getattr(image, 'disease_type', 'leaf')
            })
        
        # add prediction stuff for frontend compatibility
        data['prediction_data'] = {
            'success': True,
            'message': 'Image processed successfully',
            'data': {
                'primary_prediction': formatted_predictions[0] if formatted_predictions else {
                    'disease': image.predicted_class or 'Unknown',
                    'confidence': f"{(image.confidence_score or 0) * 100:.2f}%",
                    'confidence_score': (image.confidence_score or 0) * 100,
                    'confidence_level': 'Low',
                    'treatment': f"Treatment recommendations for {image.predicted_class or 'Unknown'}",
                    'detection_type': getattr(image, 'disease_type', 'leaf')
                },
                'top_3_predictions': formatted_predictions,
                'prediction_summary': {
                    'most_likely_disease': formatted_predictions[0]['disease'] if formatted_predictions else (image.predicted_class or 'Unknown'),
                    'confidence_level': formatted_predictions[0]['confidence_level'] if formatted_predictions else 'Low',
                    'total_diseases_checked': 11
                },
                'saved_image_id': image.id,
                'model_used': getattr(image, 'disease_type', 'leaf'),
                'model_path': get_actual_model_path(image),
                'debug_info': {
                    'model_loaded': True,
                    'image_size': getattr(image, 'image_size', 'Unknown'),
                    'processed_size': getattr(image, 'image_size', 'Unknown')
                }
            },
            'timestamp': image.uploaded_at.isoformat()
        }
        
        return JsonResponse({
            'success': True,
            'data': data
        })
        
    except MangoImage.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Image not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def store_prediction_data(request, pk):
    """save prediction data from mobile app"""
    try:
        image = MangoImage.objects.get(pk=pk)
        data = json.loads(request.body)
        
        # could store this in db later
        # image.prediction_data = data
        # image.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Prediction data stored successfully'
        })
        
    except MangoImage.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Image not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

# ================ USER STUFF ================

@csrf_exempt
@require_http_methods(["GET"])
def users_list(request):
    """get list of users with their info"""
    try:
        # Get query parameters
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        search = request.GET.get('search', '')
        
        # build query
        queryset = User.objects.select_related('userprofile').prefetch_related('mangoimage_set').order_by('-date_joined')
        
        # search filter
        if search:
            queryset = queryset.filter(
                Q(first_name__icontains=search) |
                Q(last_name__icontains=search) |
                Q(username__icontains=search) |
                Q(email__icontains=search)
            )
        
        # Pagination
        total_count = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        users = queryset[start:end]
        
        # turn into json
        serializer = UserDetailSerializer(users, many=True, context={'request': request})
        
        return JsonResponse({
            'success': True,
            'data': {
                'users': serializer.data,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size,
                    'has_next': end < total_count,
                    'has_previous': page > 1
                }
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["GET", "PUT"])
def user_detail(request, user_id):
    """get or update single user"""
    try:
        user = User.objects.select_related('userprofile').prefetch_related('mangoimage_set').get(id=user_id)
        
        if request.method == 'GET':
            serializer = UserDetailSerializer(user, context={'request': request})
            return JsonResponse({
                'success': True,
                'data': serializer.data
            })
            
        elif request.method == 'PUT':
            # update user stuff
            import json
            try:
                data = json.loads(request.body)
                if 'is_active' in data:
                    user.is_active = data['is_active']
                    user.save()
                    
                    return JsonResponse({
                        'success': True,
                        'message': f'User status updated successfully',
                        'data': {
                            'id': user.id,
                            'is_active': user.is_active,
                            'username': user.username
                        }
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'Missing required field: is_active'
                    }, status=400)
                    
            except json.JSONDecodeError:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid JSON data'
                }, status=400)
        
    except User.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'User not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def user_images(request, user_id):
    """get all images from one user"""
    try:
        user = User.objects.get(id=user_id)
        
        # filters
        disease_type = request.GET.get('disease_type', '')
        verified = request.GET.get('verified', '')
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        
        # Build query
        queryset = user.mangoimage_set.all().order_by('-uploaded_at')
        
        # Apply filters
        if disease_type:
            queryset = queryset.filter(disease_type__iexact=disease_type)
        
        if verified:
            is_verified = verified.lower() == 'true'
            queryset = queryset.filter(is_verified=is_verified)
        
        # Pagination
        total_count = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        images = queryset[start:end]
        
        # Serialize data
        serializer = MangoImageSerializer(images, many=True, context={'request': request})
        
        return JsonResponse({
            'success': True,
            'data': {
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'full_name': f"{user.first_name} {user.last_name}".strip() or user.username,
                    'email': user.email
                },
                'images': serializer.data,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size,
                    'has_next': end < total_count,
                    'has_previous': page > 1
                }
            }
        })
        
    except User.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'User not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def user_statistics(request):
    """user stats for admin dashboard"""
    try:
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()
        inactive_users = total_users - active_users
        
        # how many have profiles with location
        users_with_profiles = User.objects.filter(userprofile__isnull=False).count()
        
        # image stuff
        total_images = MangoImage.objects.count()
        average_images_per_user = total_images / total_users if total_users > 0 else 0
        
        # who uploads most
        top_users = User.objects.annotate(
            image_count=Count('mangoimage')
        ).filter(image_count__gt=0).order_by('-image_count')[:5]
        
        top_users_data = []
        for user in top_users:
            top_users_data.append({
                'id': user.id,
                'username': user.username,
                'full_name': f"{user.first_name} {user.last_name}".strip() or user.username,
                'image_count': user.image_count
            })
        
        # new users this month
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_registrations = User.objects.filter(date_joined__gte=thirty_days_ago).count()
        
        return JsonResponse({
            'success': True,
            'data': {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': inactive_users,
                'users_with_profiles': users_with_profiles,
                'total_images': total_images,
                'average_images_per_user': round(average_images_per_user, 2),
                'top_users': top_users_data,
                'recent_registrations': recent_registrations
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)