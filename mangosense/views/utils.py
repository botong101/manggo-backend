from django.http import JsonResponse
from django.utils import timezone 
import json
import os
import uuid
from PIL import Image
import numpy as np

def get_client_ip(request):
    """get users ip address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def validate_password_strength(password):
    """check if password good enough"""
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if not any(char.isdigit() for char in password):
        errors.append("Password must contain at least one digit.")
    if not any(char.isupper() for char in password):
        errors.append("Password must contain at least one uppercase letter.")
    return errors

def validate_email_format(email):
    """simple email check"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# more helper stuff for mangosense

def validate_image_file(image_file):
    """make sure image is ok"""
    errors = []
    
    # max 10mb
    if image_file.size > 10 * 1024 * 1024:
        errors.append("Image size must be less than 10MB")
    
    # only these types allowed
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if image_file.content_type not in allowed_types:
        errors.append("Only JPEG, PNG, and WebP images are allowed")
    
    # check extension too
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    file_extension = image_file.name.lower().split('.')[-1]
    if f'.{file_extension}' not in allowed_extensions:
        errors.append("Invalid file extension")
    
    return errors

def get_disease_type(disease_name):
    """is it leaf or fruit disease"""
    fruit_diseases = ['Alternaria', 'Black Mould Rot', 'Stem End Rot']
    return 'fruit' if disease_name in fruit_diseases else 'leaf'

def calculate_confidence_level(confidence_score):
    """turn number into words"""
    if confidence_score >= 0.8:
        return 'High'
    elif confidence_score >= 0.6:
        return 'Medium'
    elif confidence_score >= 0.4:
        return 'Low'
    else:
        return 'Very Low'

def format_file_size(size_bytes):
    """make bytes readable like 5MB"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def sanitize_filename(filename):
    """clean up filename for storage"""
    import re
    # remove weird chars and spaces
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename

def get_prediction_summary(prediction, class_names):
    """
    Process prediction results and return a structured summary.
    
    Args:
        prediction: numpy array of prediction probabilities
        class_names: list of class names
    
    Returns:
        dict with prediction summary
    """
    try:
        # FIX: Ensure prediction is a 1D array
        prediction = np.array(prediction)
        if len(prediction.shape) > 1:
            prediction = prediction.flatten()
        
        # Get top prediction with proper integer conversion
        top_index = int(np.argmax(prediction))
        top_confidence = float(prediction[top_index]) * 100.0
        top_disease = class_names[top_index]
        
        # Get top 3 predictions
        # FIX: Use argsort properly and convert to Python list
        top_3_indices = np.argsort(prediction)[-3:][::-1]
        top_3_indices = [int(idx) for idx in top_3_indices]  # Convert to Python ints
        
        top_3_predictions = []
        for idx in top_3_indices:
            top_3_predictions.append({
                'disease': class_names[idx],
                'confidence': float(prediction[idx]) * 100.0,
                'treatment': None  # Will be filled later
            })
        
        # Determine confidence level
        confidence_level = calculate_confidence_level(top_confidence)
        
        return {
            'primary_prediction': {
                'disease': top_disease,
                'confidence': top_confidence
            },
            'top_3': top_3_predictions,
            'confidence_level': confidence_level
        }
        
    except Exception as e:
        print(f"Error in get_prediction_summary: {e}")
        import traceback
        traceback.print_exc()
        raise

def log_prediction_activity(user, image_id, prediction_result):
    """log prediction for analytics - handles anonymous users"""
    from django.utils import timezone
    import logging
    
    logger = logging.getLogger('mangosense.predictions')
    
    # Safely get user ID
    user_id = None
    if user is not None:
        try:
            if hasattr(user, 'is_authenticated') and user.is_authenticated:
                if hasattr(user, 'id'):
                    user_id = user.id
        except Exception as e:
            print(f"Error accessing user ID: {e}")
    
    log_data = {
        'user_id': user_id,
        'image_id': image_id,
        'prediction': prediction_result.get('primary_prediction', {}).get('disease'),
        'confidence': prediction_result.get('primary_prediction', {}).get('confidence'),
        'timestamp': timezone.now().isoformat(),
        'is_anonymous': user_id is None
    }
    
    logger.info(f"Prediction logged: {log_data}")
    return log_data

def validate_admin_permissions(user):
    """check if user is admin"""
    if not user or not user.is_authenticated:
        return False, "Authentication required"
    
    if not user.is_staff:
        return False, "Admin permissions required"
    
    return True, "Valid admin user"

def paginate_queryset(queryset, page_number, page_size=20):
    """split results into pages"""
    from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
    
    paginator = Paginator(queryset, page_size)
    
    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        page = paginator.page(1)
    except EmptyPage:
        page = paginator.page(paginator.num_pages)
    
    return {
        'results': page.object_list,
        'pagination': {
            'current_page': page.number,
            'total_pages': paginator.num_pages,
            'total_items': paginator.count,
            'has_next': page.has_next(),
            'has_previous': page.has_previous(),
            'next_page': page.next_page_number() if page.has_next() else None,
            'previous_page': page.previous_page_number() if page.has_previous() else None
        }
    }

def create_api_response(success=True, message='', data=None, errors=None, error_code=None):
    """
    Create a standardized API response.
    
    Args:
        success: Boolean indicating success/failure
        message: Human-readable message
        data: Response data (dict)
        errors: List of error messages
        error_code: Specific error code for frontend handling
    
    Returns:
        dict: Standardized response
    """
    from django.utils import timezone
    
    response = {
        'success': success,
        'message': message,
        'data': data or {},
        'timestamp': timezone.now().isoformat()
    }
    
    if error_code:
        response['error_code'] = error_code
    
    if errors:
        response['errors'] = errors if isinstance(errors, list) else [errors]
    
    return response

