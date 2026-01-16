from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.core.paginator import Paginator
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from ..models import MangoImage, Notification
import json
from datetime import datetime


def create_notifications_from_images():
    """make notifications from mango images"""
    from django.contrib.auth.models import User
    
    images = MangoImage.objects.filter(
        notification__isnull=True  # only ones without notifications yet
    ).order_by('-uploaded_at')
    
    # get admin user for anon uploads
    try:
        default_user = User.objects.filter(is_staff=True).first()
        if not default_user:
            # no admin? get first user or make system user
            default_user = User.objects.first()
            if not default_user:
                default_user = User.objects.create_user(
                    username='system',
                    email='system@mangosense.com',
                    first_name='System',
                    last_name='User'
                )
    except Exception:
        default_user = None
    
    created_count = 0
    for image in images:
        # use image user or default
        notification_user = image.user if image.user else default_user
        
        if notification_user:  # only if we got a user
            notification = Notification.objects.create(
                notification_type='image_upload',
                title=f'New {image.disease_type or "Mango"} Image Upload',
                message=f'{notification_user.username} uploaded a new image: {image.original_filename}' if image.user else f'Anonymous user uploaded a new image: {image.original_filename}',
                related_image=image,
                user=notification_user
            )
            created_count += 1
    
    return created_count


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def notifications_list(request):
    """
    get notifications for admin panel
    """
    try:
        # only create new ones if asked
        # prevents recreating deleted ones
        create_new = request.GET.get('create_new', 'false').lower() == 'true'
        if create_new:
            new_notifications_count = create_notifications_from_images()
        
        # get all notifications newest first
        notifications = Notification.objects.all().order_by('-created_at')
        
        # paging
        page = request.GET.get('page', 1)
        per_page = request.GET.get('per_page', 50)
        
        paginator = Paginator(notifications, per_page)
        page_notifications = paginator.get_page(page)
        
        # turn into notification data
        notifications_data = []
        for notification in page_notifications:
            image = notification.related_image
            notifications_data.append({
                'id': str(notification.id),
                'user_id': str(notification.user.id),
                'user_name': f"{notification.user.first_name} {notification.user.last_name}".strip() if (notification.user.first_name or notification.user.last_name) else notification.user.username,
                'user_email': notification.user.email,
                'image_id': str(image.id) if image else '',
                'image_name': image.original_filename if image else 'N/A',
                'timestamp': notification.created_at.isoformat(),
                'disease_classification': image.disease_classification or image.predicted_class if image else 'N/A',
                'disease_type': image.disease_type if image else 'Unknown',
                'confidence': float(image.confidence_score) if image and image.confidence_score else 0.0,
                'is_read': notification.is_read,
                'image_url': request.build_absolute_uri(image.image.url) if image and image.image else None,
                'title': notification.title,
                'message': notification.message
            })
        
        response_data = {
            'notifications': notifications_data,
            'pagination': {
                'current_page': page_notifications.number,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'has_next': page_notifications.has_next(),
                'has_previous': page_notifications.has_previous()
            }
        }
        
        return Response(response_data)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to fetch notifications: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def mark_notification_read(request, notification_id):
    """
    mark one notification as read
    """
    try:
        notification = Notification.objects.get(id=notification_id)
        notification.is_read = True
        notification.save()
        
        return Response({
            'status': 'success',
            'message': 'Notification marked as read'
        })
        
    except Notification.DoesNotExist:
        return Response(
            {'error': 'Notification not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to mark notification as read: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def mark_all_notifications_read(request):
    """
    mark all notifications read
    """
    try:
        # mark all unread as read
        updated_count = Notification.objects.filter(is_read=False).update(is_read=True)
        
        return Response({
            'status': 'success',
            'message': f'Marked {updated_count} notifications as read',
            'updated_count': updated_count
        })
        
    except Exception as e:
        return Response(
            {'error': f'Failed to mark all notifications as read: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def notification_detail(request, notification_id):
    """
    get or delete one notification
    """
    try:
        notification = Notification.objects.get(id=notification_id)
        image = notification.related_image
        
        if request.method == 'GET':
            notification_data = {
                'id': str(notification.id),
                'user_id': str(notification.user.id),
                'user_name': f"{notification.user.first_name} {notification.user.last_name}".strip() if (notification.user.first_name or notification.user.last_name) else notification.user.username,
                'user_email': notification.user.email,
                'image_id': str(image.id) if image else '',
                'image_name': image.original_filename if image else 'N/A',
                'timestamp': notification.created_at.isoformat(),
                'disease_classification': image.disease_classification or image.predicted_class if image else 'N/A',
                'disease_type': image.disease_type if image else 'Unknown',
                'confidence': float(image.confidence_score) if image and image.confidence_score else 0.0,
                'is_read': notification.is_read,
                'image_url': request.build_absolute_uri(image.image.url) if image and image.image else None,
                'is_verified': image.is_verified if image else False,
                'verified_by': image.verified_by.username if image and image.verified_by else None,
                'verified_date': image.verified_date.isoformat() if image and image.verified_date else None,
                'notes': image.notes if image else '',
                'image_size': image.image_size if image else '',
                'processing_time': image.processing_time if image else None,
                'title': notification.title,
                'message': notification.message
            }
            return Response(notification_data)
        
        elif request.method == 'DELETE':
            # save title for response
            notification_title = notification.title
            
            # delete notification but keep image
            notification.delete()
            
            return Response({
                'status': 'success', 
                'message': f'Notification "{notification_title}" deleted successfully'
            })
        
    except Notification.DoesNotExist:
        return Response(
            {'error': 'Notification not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to process notification request: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_selected_notifications(request):
    """
    delete multiple notifications at once (not the images tho)
    """
    try:
        # get ids from request
        data = json.loads(request.body) if request.body else {}
        notification_ids = data.get('ids', [])
        
        if not notification_ids:
            return Response(
                {'error': 'No notification IDs provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # get notifications to delete
        notifications = Notification.objects.filter(id__in=notification_ids)
        
        if not notifications.exists():
            return Response(
                {'error': 'No notifications found with provided IDs'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        deleted_count = 0
        deleted_titles = []
        
        # delete each one but keep images
        for notification in notifications:
            notification_title = notification.title
            deleted_titles.append(notification_title)
            
            notification.delete()
            deleted_count += 1
        
        return Response({
            'status': 'success',
            'message': f'Successfully deleted {deleted_count} notifications',
            'deleted_count': deleted_count,
            'deleted_notifications': deleted_titles
        })
        
    except json.JSONDecodeError:
        return Response(
            {'error': 'Invalid JSON in request body'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to delete selected notifications: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def mark_all_notifications_read(request):
    """
    mark all notifications read for user
    """
    try:
        # mark all unread as read
        updated_count = Notification.objects.filter(is_read=False).update(is_read=True)
        
        return Response({
            'status': 'success',
            'message': f'Marked {updated_count} notifications as read',
            'updated_count': updated_count
        })
        
    except Exception as e:
        return Response(
            {'error': f'Failed to mark all notifications as read: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        # Mark all notifications as read (or filter by user if needed)
        updated_count = Notification.objects.filter(is_read=False).update(is_read=True)
        
        return Response({
            'status': 'success',
            'message': f'Marked {updated_count} notifications as read',
            'updated_count': updated_count
        })
        
    except Exception as e:
        return Response(
            {'error': f'Failed to mark all notifications as read: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
