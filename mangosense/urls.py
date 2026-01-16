from django.urls import path
from .views import (
    # Authentication
    register_api, login_api, logout_api,
    admin_login_api, admin_refresh_token,
    
    # ML Prediction
    predict_image, test_model_status,
    
    # Admin Dashboard APIs
    disease_statistics,
    classified_images_list,
    classified_images_detail,
    image_prediction_details,
    store_prediction_data,
    bulk_update_images,
    upload_image,
    export_dataset,
)
from .views.health_views import health_check
from .views.admin_dashboard_views import (
    #user API for managemnet
    users_list,
    user_detail,
    user_images,
    user_statistics,
)
from .views.media_views import (
    #serving images
    serve_media_file,
    test_media_access,
    debug_image_url,
)
from .views.confirmation_views import (
    # confirmations
    save_user_confirmation,
    get_user_confirmations,
    get_confirmation_statistics,
)
from .views.notification_views import (
    # notif 
    notifications_list,
    mark_notification_read,
    mark_all_notifications_read,
    notification_detail,
    delete_selected_notifications,
)
from django.conf import settings
from django.conf.urls.static import static

app_name = 'mangosense'

urlpatterns = [
    
    #mobile login 
    path('register/', register_api, name='register_api'),
    path('login/', login_api, name='login_api'),
    path('logout/', logout_api, name='logout_api'),
    
    #admin login 
    path('auth/login/', admin_login_api, name='admin_login'),
    path('auth/refresh/', admin_refresh_token, name='admin_refresh'),
    
    #ai 
    path('predict/', predict_image, name='predict_image'),
    path('test-model/', test_model_status, name='test_model_status'),
    
    #admin panel
    path('disease-statistics/', disease_statistics, name='disease_statistics'),
    path('classified-images/', classified_images_list, name='classified_images_list'),
    path('classified-images/<int:pk>/', classified_images_detail, name='classified_images_detail'),
    path('classified-images/<int:pk>/prediction-details/', image_prediction_details, name='image_prediction_details'),
    path('classified-images/<int:pk>/store-prediction/', store_prediction_data, name='store_prediction_data'),
    path('classified-images/bulk-update/', bulk_update_images, name='bulk_update_images'),
    path('upload-image/', upload_image, name='upload_image'),
    path('export-dataset/', export_dataset, name='export_dataset'),
    
    #images serving
    path('media/<path:file_path>', serve_media_file, name='serve_media_file'),
    path('test-media/', test_media_access, name='test_media_access'),
    path('debug-image/<int:image_id>/', debug_image_url, name='debug_image_url'),
    
    #user feedback
    path('save-confirmation/', save_user_confirmation, name='save_user_confirmation'),
    path('user-confirmations/', get_user_confirmations, name='get_user_confirmations'),
    path('confirmation-statistics/', get_confirmation_statistics, name='get_confirmation_statistics'),
    
    #notifs
    path('notifications/', notifications_list, name='notifications_list'),
    path('notifications/<int:notification_id>/', notification_detail, name='notification_detail'),
    path('notifications/<int:notification_id>/mark-read/', mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', mark_all_notifications_read, name='mark_all_notifications_read'),
    path('notifications/delete-selected/', delete_selected_notifications, name='delete_selected_notifications'),
    
    #users admin
    path('users/', users_list, name='users_list'),
    path('users/<int:user_id>/', user_detail, name='user_detail'),
    path('users/<int:user_id>/images/', user_images, name='user_images'),
    path('users/statistics/', user_statistics, name='user_statistics'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)