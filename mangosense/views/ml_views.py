from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
from PIL import Image
import numpy as np
import os
import gc
import json
import time
import base64
import re
from io import BytesIO
from ..models import MangoImage, MLModel, PredictionLog, Notification
from .utils import (
    get_client_ip, validate_image_file, get_disease_type,
    calculate_confidence_level, get_prediction_summary,
    log_prediction_activity, 
    create_api_response
)

import tensorflow as tf

# image size for model — MUST match what the model was trained on
# all 4 models (gate leaf, gate fruit, disease leaf, disease fruit) use 224x224
IMG_SIZE = (224, 224)

# ==================== GATE MODEL CLASS NAMES ====================
# Leaf gate model - 6 classes (alphabetically ordered)
GATE_LEAF_CLASS_NAMES = [
    'Black plum',
    'Guava',
    'Jackfruit',
    'Lychee',
    'Mango',
    'Plum'
]

# Fruit gate model - 5 classes (alphabetically ordered)
GATE_FRUIT_CLASS_NAMES = [
    'Apple',
    'Banana',
    'Grape',
    'Mango',
    'Strawberry'
]

# Index for "Mango" in the class lists (0-based index)
GATE_VALID_INDEX_LEAF = 4  # "Mango" is at index 4 in the leaf gate list
GATE_VALID_INDEX_FRUIT = 3  # "Mango" is at index 3 in the fruit gate list

# Minimum gate confidence thresholds
GATE_CONFIDENCE_THRESHOLD_LEAF = 40.0  # More lenient for diseased leaves
GATE_CONFIDENCE_THRESHOLD_FRUIT = 50.0

# ==================== DISEASE MODEL CLASS NAMES ====================
# diseases the leaf model knows
LEAF_CLASS_NAMES = [
    'Anthracnose','Die Back', 'Healthy','Powdery Mildew','Sooty Mold',
]

FRUIT_CLASS_NAMES = [
    'Anthracnose', 'Healthy' 
]

# what to do for each disease
treatment_suggestions = {
    'Anthracnose': 'The diseased twigs should be pruned and burnt along with fallen leaves. Spraying twice with Carbendazim (Bavistin 0.1%) at 15 days interval during flowering controls blossom infection.',
    'Bacterial Canker': 'Three sprays of Streptocycline (0.01%) or Agrimycin-100 (0.01%) after first visual symptom at 10 day intervals are effective in controlling the disease.',
    'Cutting Weevil': 'Use recommended insecticides and remove infested plant material.',
    'Die Back': 'Pruning of the diseased twigs 2-3 inches below the affected portion and spraying Copper Oxychloride (0.3%) on infected trees controls the disease.',
    'Gall Midge': 'Remove and destroy infested fruits; use appropriate insecticides.',
    'Healthy': 'No treatment needed. Maintain good agricultural practices.',
    'Powdery Mildew': 'Alternate spraying of Wettable sulphur 0.2 per cent at 15 days interval are recommended for effective control of the disease.',
    'Sooty Mold': 'Pruning of affected branches and their prompt destruction followed by spraying of Wettasulf (0.2%) helps to control the disease.',
    'Black Mold Rot': 'Improve air circulation and apply fungicides as needed.',
    'Stem End Rot': 'Proper post-harvest handling and storage conditions are essential.'
}



def get_treatment_for_disease(disease_name):
   
    if not disease_name:
        return "No treatment information available - disease name is empty."
    
    # try exact match first
    treatment = treatment_suggestions.get(disease_name)
    if treatment:
        return treatment
    
    # try ignoring caps
    disease_lower = disease_name.lower()
    for key, value in treatment_suggestions.items():
        if key.lower() == disease_lower:
            return value
    
    # try matching with spaces instead of underscores
    disease_normalized = disease_name.replace('_', ' ').replace('-', ' ').strip()
    for key, value in treatment_suggestions.items():
        key_normalized = key.replace('_', ' ').replace('-', ' ').strip()
        if disease_normalized.lower() == key_normalized.lower():
            return value
    
    # print what we got for debugging
    available_keys = list(treatment_suggestions.keys())
    
    return f"No treatment information available for '{disease_name}'. Please consult with an agricultural expert."

# fallback filenames if DB has no config yet
_DEFAULT_LEAF_MODEL       = 'leaf-edge-model.keras'
_DEFAULT_FRUIT_MODEL      = 'fruit-mobilenetv2.keras'
_DEFAULT_GATE_LEAF_MODEL  = 'gate-leaf-model-2.keras'
_DEFAULT_GATE_FRUIT_MODEL = 'mango-fruit-vs-others.keras'


def get_active_model_path(detection_type: str, is_gate: bool = False) -> str:
    """
    Build path to the active model file.
    detection_type: 'leaf' or 'fruit'
    is_gate: True = gate validation model, False = disease classification model
    """
    if is_gate:
        config_key = f'gate_{detection_type}'
        default = _DEFAULT_GATE_LEAF_MODEL if detection_type == 'leaf' else _DEFAULT_GATE_FRUIT_MODEL
    else:
        config_key = detection_type
        default = _DEFAULT_LEAF_MODEL if detection_type == 'leaf' else _DEFAULT_FRUIT_MODEL

    try:
        from ..models import ModelConfig
        config = ModelConfig.objects.get(detection_type=config_key)
        filename = config.model_filename
    except Exception:
        filename = default

    return os.path.join(settings.BASE_DIR, 'models', filename)


def decode_base64_image(data):
    """
    Decode base64 image data to bytes.
    Handles data URLs (data:image/jpeg;base64,...) and raw base64 strings.
    """
    # Check if it's a data URL
    if isinstance(data, str):
        # Remove data URL prefix if present
        if data.startswith('data:'):
            # Extract base64 part after the comma
            match = re.match(r'data:image/[^;]+;base64,(.+)', data)
            if match:
                data = match.group(1)
        
        # Decode base64
        return base64.b64decode(data)
    return data


def preprocess_image(image_file):
    """
    Preprocessing for inference.
    
    The model now has architecture-specific preprocessing baked into
    its graph (as a Lambda layer during training), so we only need to:
      1. Resize to the expected input size
      2. Keep pixel values as float32 in [0, 255]
      3. Add batch dimension
    
    Do NOT normalize to [0,1] or use preprocess_input here —
    the model handles it internally.
    
    All 4 models use 224x224 MobileNetV2 pretrained.
    """
    try:
        file_content = None
        
        # Handle different input types
        if hasattr(image_file, 'read'):
            # It's a file-like object (InMemoryUploadedFile, etc.)
            image_file.seek(0)
            file_content = image_file.read()
            print(f"Read {len(file_content)} bytes from file-like object")
        elif isinstance(image_file, bytes):
            file_content = image_file
            print(f"Received {len(file_content)} raw bytes")
        elif isinstance(image_file, str):
            # Could be base64 encoded or a file path
            if os.path.exists(image_file):
                with open(image_file, 'rb') as f:
                    file_content = f.read()
                print(f"Read {len(file_content)} bytes from file path")
            else:
                # Assume base64
                file_content = decode_base64_image(image_file)
                print(f"Decoded {len(file_content)} bytes from base64")
        else:
            raise Exception(f"Unsupported image input type: {type(image_file)}")
        
        # Verify file is not empty
        if not file_content:
            raise Exception("Uploaded file is empty")
        
        # Check for base64-encoded content in binary data
        # Sometimes the frontend sends base64 as a string inside the file
        if file_content[:5] == b'data:' or file_content[:20].startswith(b'data:image'):
            print("Detected base64 data URL in file content")
            file_content = decode_base64_image(file_content.decode('utf-8'))
        
        # Log first few bytes for debugging
        print(f"First 20 bytes: {file_content[:20]}")
        
        # Check for common image magic bytes
        is_jpeg = file_content[:2] == b'\xff\xd8'
        is_png = file_content[:8] == b'\x89PNG\r\n\x1a\n'
        is_gif = file_content[:6] in (b'GIF87a', b'GIF89a')
        is_webp = file_content[:4] == b'RIFF' and file_content[8:12] == b'WEBP'
        print(f"Image format detection - JPEG: {is_jpeg}, PNG: {is_png}, GIF: {is_gif}, WebP: {is_webp}")
        
        if not (is_jpeg or is_png or is_gif or is_webp):
            # Try to decode as base64 if it looks like text
            try:
                decoded = base64.b64decode(file_content)
                if decoded[:2] == b'\xff\xd8' or decoded[:8] == b'\x89PNG\r\n\x1a\n':
                    print("Successfully decoded raw base64 to image bytes")
                    file_content = decoded
            except Exception:
                pass  # Not base64, continue with original content
        
        # Create BytesIO and open with PIL
        image_bytes = BytesIO(file_content)
        img = Image.open(image_bytes)
        
        # Convert to RGB (handles PNG, RGBA, etc.)
        img = img.convert('RGB')
        original_size = img.size
        print(f"Successfully opened image: {original_size}")
        
        # Resize to model input size
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array [0, 255] float32
        img_array = np.array(img).astype("float32")
        
        # Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_size
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        print(f"Image file type: {type(image_file)}")
        print(f"Image file name: {getattr(image_file, 'name', 'unknown')}")
        if hasattr(image_file, 'size'):
            print(f"Image file size: {image_file.size} bytes")
        if file_content:
            print(f"Content length: {len(file_content)} bytes")
            print(f"Content preview: {file_content[:100]}")
        raise Exception(f"Failed to preprocess image: {str(e)}")





@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])  # Allow unauthenticated predictions
def predict_image(request):
    
    import time
    start_time = time.time()
    
    # debug
    if 'image' not in request.FILES:
        return JsonResponse(
            create_api_response(
                success=False,
                message='No image uploaded',
                errors=['Image file is required']
            ),
            status=400
        )

    try:
        image_file = request.FILES['image']
        
        # === DEBUG: Log exactly what we received ===
        print("=" * 50)
        print("IMAGE FILE DEBUG INFO:")
        print(f"  Name: {image_file.name}")
        print(f"  Size: {image_file.size} bytes")
        print(f"  Content-Type: {image_file.content_type}")
        print(f"  Type: {type(image_file)}")
        
        # Read first 50 bytes to check what we actually got
        image_file.seek(0)
        first_bytes = image_file.read(50)
        print(f"  First 50 bytes (raw): {first_bytes[:50]}")
        print(f"  First 50 bytes (hex): {first_bytes[:50].hex()}")
        image_file.seek(0)  # Reset for actual processing
        
        # Check if this looks like valid image data
        is_jpeg = first_bytes[:2] == b'\xff\xd8'
        is_png = first_bytes[:8] == b'\x89PNG\r\n\x1a\n'
        print(f"  Looks like JPEG: {is_jpeg}")
        print(f"  Looks like PNG: {is_png}")
        print("=" * 50)
        # === END DEBUG ===
        
        # get gps data from request
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        location_accuracy_confirmed = request.data.get('location_accuracy_confirmed', 'false').lower() == 'true'
        location_source = request.data.get('location_source', '')
        location_address = request.data.get('location_address', '')
        
        # check if just preview (dont save to db)
        preview_only = request.data.get('preview_only', 'false').lower() == 'true'
        
        # get what user said about detection
        is_detection_correct = request.data.get('is_detection_correct', '').lower() == 'true'
        user_feedback = request.data.get('user_feedback', '')
        
        # get symptoms they picked
        try:
            selected_symptoms = json.loads(request.data.get('selected_symptoms', '[]')) if request.data.get('selected_symptoms') else []
        except (json.JSONDecodeError, TypeError):
            selected_symptoms = []
            
        try:
            primary_symptoms = json.loads(request.data.get('primary_symptoms', '[]')) if request.data.get('primary_symptoms') else []
        except (json.JSONDecodeError, TypeError):
            primary_symptoms = []
            
        try:
            alternative_symptoms = json.loads(request.data.get('alternative_symptoms', '[]')) if request.data.get('alternative_symptoms') else []
        except (json.JSONDecodeError, TypeError):
            alternative_symptoms = []
            
        detected_disease = request.data.get('detected_disease', '')
        
        try:
            top_diseases = json.loads(request.data.get('top_diseases', '[]')) if request.data.get('top_diseases') else []
        except (json.JSONDecodeError, TypeError):
            top_diseases = []
            
        try:
            symptoms_data = json.loads(request.data.get('symptoms_data', '{}')) if request.data.get('symptoms_data') else {}
        except (json.JSONDecodeError, TypeError):
            symptoms_data = {}
        
        # make sure image is ok
        validation_errors = validate_image_file(image_file)
        if validation_errors:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Invalid image file',
                    errors=validation_errors
                ),
                status=400
            )

        # prep the image — same 224x224 for gate and disease models
        try:
            img_array, original_size = preprocess_image(image_file)
        except Exception as preprocessing_error:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Image preprocessing failed',
                    errors=[str(preprocessing_error)]
                ),
                status=500
            )

        # fruit or leaf?
        detection_type = request.data.get('detection_type', 'leaf')
        
        # gps stuff again
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        location_accuracy_confirmed = request.data.get('location_accuracy_confirmed', 'false').lower() == 'true'
        location_source = request.data.get('location_source', '')
        location_address = request.data.get('location_address', '')

        # ================================================================
        #  STAGE 1 — GATE MODEL (is this a mango leaf/fruit?)
        # ================================================================
        gate_model_path = get_active_model_path(detection_type, is_gate=True)

        # defaults — if gate model missing or breaks, let image through
        gate_passed = True
        gate_confidence = None
        gate_prediction_label = None

        if os.path.exists(gate_model_path):
            try:
                gate_model = tf.keras.models.load_model(gate_model_path)
                gate_pred = gate_model.predict(img_array)
                gate_pred = np.array(gate_pred).flatten()

                if detection_type == 'fruit':
                    valid_idx = GATE_VALID_INDEX_FRUIT
                    gate_cls = GATE_FRUIT_CLASS_NAMES
                else:
                    valid_idx = GATE_VALID_INDEX_LEAF
                    gate_cls = GATE_LEAF_CLASS_NAMES

                gate_confidence = float(gate_pred[valid_idx]) * 100
                gate_predicted_idx = int(np.argmax(gate_pred))
                gate_prediction_label = gate_cls[gate_predicted_idx]

                threshold = GATE_CONFIDENCE_THRESHOLD_FRUIT if detection_type == 'fruit' else GATE_CONFIDENCE_THRESHOLD_LEAF
                gate_passed = (
                    gate_predicted_idx == valid_idx
                    and gate_confidence >= threshold
                )

                # free memory
                del gate_model
                gc.collect()

            except Exception as gate_err:
                print(f"Gate model error: {gate_err} — skipping validation")
        else:
            print(f"Gate model not found at {gate_model_path} — skipping validation")

        # ---- gate rejected → return early ----
        if not gate_passed:
            part = detection_type.capitalize()  # "Leaf" or "Fruit"
            return JsonResponse(
                create_api_response(
                    success=True,
                    data={
                        'primary_prediction': {
                            'disease': f'Not a Mango {part}',
                            'confidence': f"{gate_confidence:.2f}%",
                            'confidence_score': gate_confidence or 0,
                            'confidence_level': 'Low',
                            'treatment': (
                                f"The uploaded image does not appear to be a mango {detection_type}. "
                                f"Please upload a clear image of a mango {detection_type} and try again."
                            ),
                            'detection_type': detection_type
                        },
                        'top_3_predictions': [],
                        'prediction_summary': {
                            'most_likely': f'Not a Mango {part}',
                            'confidence_level': 'Low',
                            'total_diseases_checked': 0
                        },
                        'alternative_symptoms': {
                            'primary_disease': f'Not a Mango {part}',
                            'primary_disease_symptoms': [],
                            'alternative_diseases': []
                        },
                        'user_verification': {
                            'selected_symptoms': [],
                            'primary_symptoms': [],
                            'alternative_symptoms': [],
                            'detected_disease': f'Not a Mango {part}',
                            'is_detection_correct': False,
                            'user_feedback': ''
                        },
                        'gate_validation': {
                            'passed': False,
                            'gate_prediction': gate_prediction_label,
                            'gate_confidence': gate_confidence,
                            'message': f'Image classified as "{gate_prediction_label}" by gate model'
                        },
                        'saved_image_id': None,
                        'model_used': detection_type,
                        'debug_info': {
                            'gate_model_used': True,
                            'gate_model_path': gate_model_path,
                            'processing_time': time.time() - start_time,
                            'image_size': original_size,
                            'processed_size': IMG_SIZE
                        }
                    },
                    message=(
                        f'The uploaded image does not appear to be a mango {detection_type}. '
                        f'Please upload a clear image of a mango {detection_type}.'
                    )
                )
            )

        # ================================================================
        #  STAGE 2 — DISEASE CLASSIFICATION (existing logic)
        # ================================================================
        
        # pick which model to use
        if detection_type == 'fruit':
            model_path = get_active_model_path('fruit')
            model_used = 'fruit'
            model_class_names = FRUIT_CLASS_NAMES
        else:
            model_path = get_active_model_path('leaf')
            model_used = 'leaf'
            model_class_names = LEAF_CLASS_NAMES


        # check model file exists
        if not os.path.exists(model_path):
            return JsonResponse(
                create_api_response(
                    success=False,
                    message=f'Model file not found: {model_used}',
                    errors=[f'Model file {model_path} does not exist']
                ),
                status=500
            )

        # load the ai model
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as model_error:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Failed to load ML model',
                    errors=[str(model_error)]
                ),
                status=500
            )

        # run it thru the model
        try:
            prediction = model.predict(img_array)
            prediction = np.array(prediction).flatten()
        except Exception as prediction_error:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='ML prediction failed',
                    errors=[str(prediction_error)]
                ),
                status=500
            )

        # organize the results
        prediction_summary = get_prediction_summary(prediction, model_class_names)

        # min confidence to show disease
        CONFIDENCE_THRESHOLD = 20.0

        # if too low just say unknown
        if prediction_summary['primary_prediction']['confidence'] < CONFIDENCE_THRESHOLD:
            unknown_response = {
                'disease': 'Unknown',
                'confidence': f"{prediction_summary['primary_prediction']['confidence']:.2f}%",
                'confidence_score': prediction_summary['primary_prediction']['confidence'],
                'confidence_level': 'Low',
                'treatment': "The uploaded image could not be confidently classified. Please ensure the image is of a mango leaf or fruit and try again.",
                'detection_type': model_used
            }
            response_data = {
                'primary_prediction': unknown_response,
                'top_3_predictions': [],
                'prediction_summary': {
                    'most_likely': 'Unknown',
                    'confidence_level': 'Low',
                    'total_diseases_checked': len(model_class_names)
                },
                'alternative_symptoms': {
                    'primary_disease': 'Unknown',
                    'primary_disease_symptoms': [],
                    'alternative_diseases': []
                },
                'user_verification': {
                    'selected_symptoms': [],
                    'primary_symptoms': [],
                    'alternative_symptoms': [],
                    'detected_disease': 'Unknown',
                    'is_detection_correct': False,
                    'user_feedback': ''
                },
                'gate_validation': {
                    'passed': True,
                    'gate_prediction': gate_prediction_label,
                    'gate_confidence': gate_confidence,
                    'message': f'Image validated as mango {detection_type}'
                },
                'saved_image_id': None,
                'model_used': model_used,
                'model_path': model_path,
                'debug_info': {
                    'gate_model_used': gate_model_path if os.path.exists(gate_model_path) else None,
                    'model_loaded': True,
                    'image_size': original_size,
                    'processed_size': IMG_SIZE
                }
            }
            return JsonResponse(
                create_api_response(
                    success=True,
                    data=response_data,
                    message='Could not confidently classify the image. Please upload a clear image of a mango leaf or fruit.'
                )
            )

        # add treatment info
        for pred in prediction_summary['top_3']:
            pred['treatment'] = get_treatment_for_disease(pred['disease'])
            pred['detection_type'] = model_used

        # frontend will handle symptoms itself
        # it has getDiseaseSymptoms() for that
        primary_disease = prediction_summary['primary_prediction']['disease']
        alternative_diseases = [pred['disease'] for pred in prediction_summary['top_3'][1:3]]  # top 2-3 diseases

        # save to db unless just preview
        saved_image_id = None
        if not preview_only:
            try:
                image_file.seek(0)
                
                # set up location data for saving
                location_data = {}
                if latitude and longitude:
                    try:
                        location_data.update({
                            'latitude': float(latitude),
                            'longitude': float(longitude),
                            'location_consent_given': True,  # Consent was given during registration
                            'location_accuracy_confirmed': location_accuracy_confirmed,
                            'location_source': location_source,
                            'location_address': location_address,
                        })
                    except (ValueError, TypeError) as e:
                        location_data.update({
                            'location_consent_given': False,
                            'location_accuracy_confirmed': False,
                        })
                else:
                    location_data.update({
                        'location_consent_given': False,
                        'location_accuracy_confirmed': False,
                    })
                
                # how long did it take
                processing_time = time.time() - start_time
                
                # Safely get the user for saving
                image_user = None
                if hasattr(request, 'user'):
                    if hasattr(request.user, 'is_authenticated') and request.user.is_authenticated:
                        try:
                            # Ensure user actually exists in database
                            if request.user.id is not None:
                                image_user = request.user
                        except Exception as e:
                            print(f"Error accessing user for image save: {e}")
                
                mango_image = MangoImage.objects.create(
                    image=image_file,
                    original_filename=image_file.name,
                    predicted_class=prediction_summary['primary_prediction']['disease'],
                    disease_classification=prediction_summary['primary_prediction']['disease'],
                    disease_type=model_used, 
                    model_used=model_used,  # Store which model was actually used
                    model_filename=os.path.basename(model_path),  # Store the actual model filename
                    confidence_score=prediction_summary['primary_prediction']['confidence'] / 100,
                    user=image_user,  # Use safely retrieved user
                    image_size=f"{original_size[0]}x{original_size[1]}",
                    processing_time=processing_time,
                    notes=f"Predicted via mobile app with {prediction_summary['primary_prediction']['confidence']:.2f}% confidence",
                    is_verified=False,  # Always default to unverified - admin must manually verify
                    user_feedback=user_feedback if user_feedback else None,  # User feedback can be NULL
                    user_confirmed_correct=is_detection_correct if user_feedback else None,  # Save user confirmation decision
                    # Add symptoms data
                    selected_symptoms=selected_symptoms if selected_symptoms else None,
                    primary_symptoms=primary_symptoms if primary_symptoms else None,
                    alternative_symptoms=alternative_symptoms if alternative_symptoms else None,
                    detected_disease=detected_disease if detected_disease else prediction_summary['primary_prediction']['disease'],
                    top_diseases=top_diseases if top_diseases else None,
                    symptoms_data=symptoms_data if symptoms_data else None,
                    **location_data  # Add all location data
                )
                
                # Log prediction activity with proper error handling
                try:
                    current_user = None
                    if hasattr(request, 'user'):
                        if hasattr(request.user, 'is_authenticated') and request.user.is_authenticated:
                            current_user = request.user
                    log_prediction_activity(current_user, mango_image.id, prediction_summary)
                except Exception as log_error:
                    print(f"Failed to log prediction activity: {log_error}")
                    # Continue anyway - logging shouldn't break predictions
                
                saved_image_id = mango_image.id
                
                # make notif for admin
                try:
                    # get user who uploaded or find an admin
                    notification_user = mango_image.user if mango_image.user else None
                    
                    # if nobody logged in grab admin for notif
                    if not notification_user:
                        from django.contrib.auth.models import User
                        notification_user = User.objects.filter(is_staff=True).first()
                    
                    if notification_user:
                        # make the notification
                        Notification.objects.create(
                            notification_type='image_upload',
                            title=f'New {model_used.title()} Image Upload',
                            message=f'A new {model_used} image "{mango_image.original_filename}" was uploaded and classified as {prediction_summary["primary_prediction"]["disease"]} with {prediction_summary["primary_prediction"]["confidence"]:.1f}% confidence.',
                            related_image=mango_image,
                            user=notification_user
                        )
                    else:
                        print(f"No user available for notification creation")
                except Exception as notification_error:
                    print(f"Error creating notification: {notification_error}")
                    # dont break everything if notif fails
            except Exception as e:
                print(f"Error saving image to database: {e}")
                saved_image_id = None

        # clean up memory
        gc.collect()

        response_data = {
            'primary_prediction': {
                'disease': prediction_summary['primary_prediction']['disease'],
                'confidence': f"{prediction_summary['primary_prediction']['confidence']:.2f}%",
                'confidence_score': prediction_summary['primary_prediction']['confidence'],
                'confidence_level': prediction_summary['confidence_level'],
                'treatment': get_treatment_for_disease(prediction_summary['primary_prediction']['disease']),
                'detection_type': model_used
            },
            'top_3_predictions': prediction_summary['top_3'],
            'prediction_summary': {
                'most_likely': prediction_summary['primary_prediction']['disease'],
                'confidence_level': prediction_summary['confidence_level'],
                'total_diseases_checked': len(model_class_names)
            },
            'alternative_symptoms': {
                'primary_disease': primary_disease,
                'primary_disease_symptoms': [],  # Frontend will generate using getDiseaseSymptoms()
                'alternative_diseases': alternative_diseases  # Just disease names, frontend will get symptoms
            },
            'user_verification': {
                'selected_symptoms': selected_symptoms,
                'primary_symptoms': primary_symptoms,
                'alternative_symptoms': alternative_symptoms,
                'detected_disease': detected_disease,
                'is_detection_correct': is_detection_correct,
                'user_feedback': user_feedback
            },
            # gate validation info for frontend
            'gate_validation': {
                'passed': True,
                'gate_prediction': gate_prediction_label,
                'gate_confidence': gate_confidence,
                'message': f'Image validated as mango {detection_type}'
            },
            'model_used': model_used,
            'model_path': model_path,
            'debug_info': {
                'gate_model_used': gate_model_path if os.path.exists(gate_model_path) else None,
                'model_loaded': True,
                'image_size': original_size,
                'processed_size': IMG_SIZE
            }
        }
        
        # only add image id if we actually saved it
        if not preview_only and saved_image_id:
            response_data['saved_image_id'] = saved_image_id
        try:
            probs_list = prediction.tolist() if hasattr(prediction, 'tolist') else list(map(float, prediction))
            labels_list = model_class_names
            response_time = time.time() - start_time
            PredictionLog.objects.create(
                image=mango_image if 'mango_image' in locals() else None,
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                response_time=response_time,
                probabilities=probs_list,
                labels=labels_list,
                prediction_summary=prediction_summary,
                raw_response=response_data
            )
        except Exception as e:
            print(f"Failed to log prediction activity: {str(e)}")

        return JsonResponse(
            create_api_response(
                success=True,
                data=response_data,
                message='Image processed successfully'
            )
        )

    except Exception as e:
        return JsonResponse(
            create_api_response(
                success=False,
                message='Prediction failed',
                errors=[str(e)]
            ),
            status=500
        )





@api_view(['GET'])
def test_model_status(request):
    
    try:
        # see if theres an active model in db
        active_model = MLModel.objects.filter(is_active=True).first()
        
        leaf_model_path = get_active_model_path('leaf')
        fruit_model_path = get_active_model_path('fruit')
        gate_leaf_model_path = get_active_model_path('leaf', is_gate=True)
        gate_fruit_model_path = get_active_model_path('fruit', is_gate=True)
        
        model_status = {
            'model_loaded': active_model is not None,
            'leaf_model_path': leaf_model_path,
            'fruit_model_path': fruit_model_path,
            'gate_leaf_model_path': gate_leaf_model_path,
            'gate_fruit_model_path': gate_fruit_model_path,
            'leaf_model_exists': os.path.exists(leaf_model_path),
            'fruit_model_exists': os.path.exists(fruit_model_path),
            'gate_leaf_model_exists': os.path.exists(gate_leaf_model_path),
            'gate_fruit_model_exists': os.path.exists(gate_fruit_model_path),
            'leaf_class_names': LEAF_CLASS_NAMES,
            'fruit_class_names': FRUIT_CLASS_NAMES,
            'gate_leaf_class_names': GATE_LEAF_CLASS_NAMES,
            'gate_fruit_class_names': GATE_FRUIT_CLASS_NAMES,
            'leaf_classes_count': len(LEAF_CLASS_NAMES),
            'fruit_classes_count': len(FRUIT_CLASS_NAMES),
            'treatment_suggestions_count': len(treatment_suggestions),
            'gate_confidence_threshold_leaf': GATE_CONFIDENCE_THRESHOLD_LEAF,
            'gate_confidence_threshold_fruit': GATE_CONFIDENCE_THRESHOLD_FRUIT,
            'active_model': {
                'name': active_model.name if active_model else None,
                'version': active_model.version if active_model else None,
                'file_path': active_model.file_path if active_model else None,
            } if active_model else None,
            'img_size': IMG_SIZE
        }
        
        database_stats = {
            'total_images': MangoImage.objects.count(),
            'healthy_images': MangoImage.objects.filter(disease_classification='Healthy').count(),
            'diseased_images': MangoImage.objects.exclude(disease_classification='Healthy').count(),
            'verified_images': MangoImage.objects.filter(is_verified=True).count()
        }
        
        return JsonResponse(
            create_api_response(
                success=True,
                data={
                    'model_status': model_status,
                    'available_leaf_diseases': LEAF_CLASS_NAMES,
                    'available_fruit_diseases': FRUIT_CLASS_NAMES,
                    'database_stats': database_stats
                },
                message='Model status retrieved successfully'
            )
        )
        
    except Exception as e:
        return JsonResponse(
            create_api_response(
                success=False,
                message='Failed to get model status',
                errors=[str(e)]
            ),
            status=500
        )