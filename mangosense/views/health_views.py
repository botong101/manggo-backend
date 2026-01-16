from django.http import JsonResponse
from django.db import connection
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """
    check if server is alive for railway
    returns 200 if good 503 if broken
    
    checks:
    - db connection
    - models work
    - app responding
    """
    try:
        # test db connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        # check if models load ok
        from ..models import MangoImage
        
        # get some stats
        try:
            image_count = MangoImage.objects.count()
        except Exception:
            image_count = 0
        
        return JsonResponse({
            'status': 'healthy',
            'service': 'mangosense-backend',
            'database': 'connected',
            'models': 'accessible',
            'image_count': image_count,
            'timestamp': timezone.now().isoformat(),
            'version': '1.0.0'
        }, status=200)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e),
            'service': 'mangosense-backend',
            'timestamp': timezone.now().isoformat()
        }, status=503)