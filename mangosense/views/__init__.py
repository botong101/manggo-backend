from .auth_views import register_api, login_api, logout_api
from .admin_auth_views import admin_login_api, admin_refresh_token
from .ml_views import predict_image, test_model_status
from .admin_dashboard_views import (
    disease_statistics,
    classified_images_list,
    classified_images_detail,
    image_prediction_details,
    store_prediction_data,
    bulk_update_images,
    upload_image,
    export_dataset
)
from .media_views import (
    serve_media_file,
    test_media_access,
    debug_image_url
)
from .history_views import user_analysis_history
from .utils import (
    get_client_ip,
    validate_password_strength,
    validate_email_format,
    validate_image_file,
    get_disease_type,
    calculate_confidence_level,
    create_api_response
)

__all__ = [
    # Auth views
    'register_api',
    'login_api',
    'logout_api',
    'admin_login_api',
    'admin_refresh_token',
    
    # ML views
    'predict_image',
    'test_model_status',
    
    # Admin dashboard views
    'disease_statistics',
    'classified_images_list',
    'classified_images_detail',
    'bulk_update_images',
    'upload_image',
    'export_dataset',
    
    # Media views
    'serve_media_file',
    'test_media_access',
    'debug_image_url',

    # User history
    'user_analysis_history',
    
    # Utils
    'get_client_ip',
    'validate_password_strength',
    'validate_email_format',
    'validate_image_file',
    'get_disease_type',
    'calculate_confidence_level',
    'create_api_response'
]