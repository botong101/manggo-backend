"""
media file serving — redirects to Supabase S3 storage
"""
from django.conf import settings
from django.http import HttpResponseRedirect, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


@csrf_exempt
@require_http_methods(["GET"])
def serve_media_file(request, file_path):
    # files now live on Supabase Storage — redirect to the public S3 URL
    s3_url = f"{settings.MEDIA_URL}{file_path}"
    return HttpResponseRedirect(s3_url)


@csrf_exempt
@require_http_methods(["GET"])
def test_media_access(request):
    try:
        from ..models import MangoImage
        sample_images = MangoImage.objects.exclude(image='').order_by('-uploaded_at')[:5]
        sample = [
            {
                'filename': img.original_filename,
                'url': img.image.url if img.image else None,
            }
            for img in sample_images
        ]

        return JsonResponse({
            'success': True,
            'data': {
                'media_url': settings.MEDIA_URL,
                'storage_backend': settings.DEFAULT_FILE_STORAGE,
                'sample_images': sample,
                'instructions': {
                    'message': 'Files are served from Supabase Storage. Use the url field directly.',
                }
            }
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error checking media access: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def debug_image_url(request, image_id):
    """debug image url — S3-safe version"""
    try:
        from ..models import MangoImage
        from ..serializers import MangoImageSerializer

        image = MangoImage.objects.get(id=image_id)
        serializer = MangoImageSerializer(image, context={'request': request})

        return JsonResponse({
            'success': True,
            'data': {
                'image_id': image_id,
                'serialized_data': serializer.data,
                'storage_url': image.image.url if image.image else None,
                'image_name': image.image.name if image.image else None,
            }
        })

    except MangoImage.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Image not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error debugging image: {str(e)}'
        }, status=500)
