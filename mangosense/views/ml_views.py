from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gc
import time
import json

from ..models import MangoImage, MLModel, PredictionLog, Notification
from .utils import (
    get_client_ip, validate_image_file, get_disease_type,
    calculate_confidence_level, get_prediction_summary,
    log_prediction_activity, 
    create_api_response
)

# image size for model — MUST match what the model was trained on
# all 4 models (gate leaf, gate fruit, disease leaf, disease fruit) use 224x224
IMG_SIZE = (224, 224)

# ==================== GATE MODEL CLASS NAMES ====================
# these must match the class_names from your gate model training
# adjust to match your actual training folder names!
GATE_LEAF_CLASS_NAMES = [
    'Black plum',
    'Guava',
    'Jackfruit',
    'Lychee',
    'Mango',
    'Plum',
]

GATE_FRUIT_CLASS_NAMES = [
    'Apple',
    'Banana',
    'Grape',
    'Mango',    
    'Strawberry'
]

# index that means "valid mango" — adjust based on your training class order
GATE_VALID_INDEX_LEAF = 4  # "Mango" is at index 4 (0-based)
GATE_VALID_INDEX_FRUIT = 3  # "Mango" is at index 3

# minimum gate confidence to let the image through
GATE_CONFIDENCE_THRESHOLD_LEAF = 40.0  # Lower threshold for diseased leaves
GATE_CONFIDENCE_THRESHOLD_FRUIT = 50.0  # Can be stricter for fruits

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


def preprocess_image(image_file):
    """
    Preprocessing for inference.
    
    Returns:
        tuple: (img_array, original_size)
    """
    try:
        img = Image.open(image_file).convert('RGB')
        original_size = img.size
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array as float32 in [0, 255]
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Preprocessed image shape: {img_array.shape}")
        print(f"Preprocessed image dtype: {img_array.dtype}")
        print(f"Preprocessed image range: [{img_array.min()}, {img_array.max()}]")
        
        return img_array, original_size
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        raise





@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])  # Add this if endpoint should work without auth
def predict_image(request):
    start_time = time.time()
    
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
        
        # get request data
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        location_accuracy_confirmed = request.data.get('location_accuracy_confirmed', 'false').lower() == 'true'
        location_source = request.data.get('location_source', '')
        location_address = request.data.get('location_address', '')
        preview_only = request.data.get('preview_only', 'false').lower() == 'true'
        is_detection_correct = request.data.get('is_detection_correct', '').lower() == 'true'
        user_feedback = request.data.get('user_feedback', '')
        
        try:
            selected_symptoms = json.loads(request.data.get('selected_symptoms', '[]'))
        except (json.JSONDecodeError, TypeError):
            selected_symptoms = []
            
        try:
            primary_symptoms = json.loads(request.data.get('primary_symptoms', '[]'))
        except (json.JSONDecodeError, TypeError):
            primary_symptoms = []
            
        try:
            alternative_symptoms = json.loads(request.data.get('alternative_symptoms', '[]'))
        except (json.JSONDecodeError, TypeError):
            alternative_symptoms = []
            
        detected_disease = request.data.get('detected_disease', '')
        
        try:
            plant_part_affected = json.loads(request.data.get('plant_part_affected', '[]'))
        except (json.JSONDecodeError, TypeError):
            plant_part_affected = []
            
        try:
            environmental_factors = json.loads(request.data.get('environmental_factors', '[]'))
        except (json.JSONDecodeError, TypeError):
            environmental_factors = []

        # validate image
        validation_errors = validate_image_file(image_file)
        if validation_errors:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Image validation failed',
                    errors=validation_errors
                ),
                status=400
            )

        # preprocess image
        try:
            img_array, original_size = preprocess_image(image_file)
        except Exception as preprocessing_error:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Image preprocessing failed',
                    errors=[str(preprocessing_error)]
                ),
                status=400
            )

        # detection type
        detection_type = request.data.get('detection_type', 'leaf')

        # ================================================================
        #  STAGE 1 — GATE MODEL
        # ================================================================
        gate_model_path = get_active_model_path(detection_type, is_gate=True)
        gate_passed = True
        gate_confidence = None
        gate_prediction_label = None
        gate_predicted_class = None
        mango_confidence = None

        if os.path.exists(gate_model_path):
            try:
                gate_model = tf.keras.models.load_model(gate_model_path)
                gate_pred = gate_model.predict(img_array, verbose=0)
                
                # FIX: Ensure gate_pred is properly flattened
                gate_pred = np.array(gate_pred)
                if len(gate_pred.shape) > 1:
                    gate_pred = gate_pred[0]  # Get first batch item
                gate_pred = gate_pred.flatten()  # Flatten to 1D array

                if detection_type == 'fruit':
                    valid_idx = GATE_VALID_INDEX_FRUIT
                    gate_cls = GATE_FRUIT_CLASS_NAMES
                    threshold = GATE_CONFIDENCE_THRESHOLD_FRUIT
                else:
                    valid_idx = GATE_VALID_INDEX_LEAF
                    gate_cls = GATE_LEAF_CLASS_NAMES
                    threshold = GATE_CONFIDENCE_THRESHOLD_LEAF

                # FIX: Ensure we get integer index, not array
                gate_predicted_idx = int(np.argmax(gate_pred))
                
                # FIX: Check array bounds before accessing
                if gate_predicted_idx >= len(gate_cls):
                    print(f"Warning: predicted index {gate_predicted_idx} out of bounds for {len(gate_cls)} classes")
                    gate_predicted_idx = 0
                    
                gate_prediction_label = gate_cls[gate_predicted_idx]
                gate_confidence = float(gate_pred[gate_predicted_idx]) * 100.0
                
                # FIX: Check valid_idx is within bounds
                if valid_idx >= len(gate_pred):
                    print(f"Warning: valid index {valid_idx} out of bounds for prediction array of length {len(gate_pred)}")
                    mango_confidence = 0.0
                else:
                    mango_confidence = float(gate_pred[valid_idx]) * 100.0

                # Check both predicted class AND mango confidence
                if gate_predicted_idx != valid_idx or mango_confidence < threshold:
                    gate_passed = False
                    
                    if gate_predicted_idx != valid_idx:
                        print(f"❌ Gate failed: Predicted '{gate_prediction_label}' (not Mango)")
                    if mango_confidence < threshold:
                        print(f"❌ Gate failed: Mango confidence {mango_confidence:.2f}% < {threshold}%")
                else:
                    print(f"✅ Gate passed: {gate_prediction_label} @ {gate_confidence:.2f}%, Mango @ {mango_confidence:.2f}%")

                del gate_model
                gc.collect()

            except Exception as gate_err:
                print(f"Gate model error: {gate_err}")
                import traceback
                traceback.print_exc()
                gate_passed = True
        else:
            print(f"Gate model not found at {gate_model_path}")

        # Return if gate failed
        if not gate_passed:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message=f'Image rejected: Not a valid mango {detection_type}',
                    data={
                        'gate_validation': {
                            'passed': False,
                            'predicted_class': gate_prediction_label,
                            'predicted_confidence': f"{gate_confidence:.2f}%" if gate_confidence else None,
                            'mango_confidence': f"{mango_confidence:.2f}%" if mango_confidence else None,
                            'required_confidence': f"{threshold:.0f}%",
                            'message': f'Predicted as "{gate_prediction_label}" with {gate_confidence:.2f}% confidence. Mango confidence: {mango_confidence:.2f}% (required: ≥{threshold:.0f}%)'
                        },
                        'detection_type': detection_type
                    }
                ),
                status=400
            )

        # ================================================================
        #  STAGE 2 — DISEASE CLASSIFICATION
        # ================================================================
        
        if detection_type == 'fruit':
            model_path = get_active_model_path('fruit')
            model_class_names = FRUIT_CLASS_NAMES
            model_used = 'fruit'
        else:
            model_path = get_active_model_path('leaf')
            model_class_names = LEAF_CLASS_NAMES
            model_used = 'leaf'

        if not os.path.exists(model_path):
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Model file not found',
                    errors=[f'Model path: {model_path}']
                ),
                status=500
            )

        try:
            disease_model = tf.keras.models.load_model(model_path)
        except Exception as model_error:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Failed to load model',
                    errors=[str(model_error)]
                ),
                status=500
            )

        try:
            prediction = disease_model.predict(img_array, verbose=0)
            
            # FIX: Debug the prediction shape
            print(f"Raw prediction shape: {prediction.shape}")
            print(f"Raw prediction type: {type(prediction)}")
            
            # FIX: Ensure prediction is 1D array of probabilities
            prediction = np.array(prediction)
            if len(prediction.shape) > 1:
                if prediction.shape[0] == 1:
                    prediction = prediction[0]  # Remove batch dimension
                else:
                    prediction = prediction.flatten()
            
            print(f"Processed prediction shape: {prediction.shape}")
            print(f"Prediction values: {prediction}")
            print(f"Number of classes: {len(model_class_names)}")
            
            # Validate prediction array length matches class names
            if len(prediction) != len(model_class_names):
                raise ValueError(
                    f"Prediction array length ({len(prediction)}) doesn't match "
                    f"number of classes ({len(model_class_names)})"
                )
            
            del disease_model
            gc.collect()
            
        except Exception as prediction_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Prediction error: {prediction_error}")
            print(error_trace)
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Disease prediction failed',
                    errors=[
                        str(prediction_error),
                        f"Prediction shape issue - check server logs"
                    ]
                ),
                status=500
            )

        # Get prediction summary with error handling
        try:
            # Make sure prediction is properly formatted
            prediction_array = np.array(prediction, dtype=np.float32)
            if prediction_array.ndim != 1:
                prediction_array = prediction_array.flatten()
            
            print(f"Final prediction array shape before summary: {prediction_array.shape}")
            
            prediction_summary = get_prediction_summary(prediction_array, model_class_names)
            
        except Exception as summary_error:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Summary error: {summary_error}")
            print(error_trace)
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='Failed to process prediction results',
                    errors=[
                        str(summary_error),
                        "Check prediction array format"
                    ]
                ),
                status=500
            )

        # Set confidence threshold
        CONFIDENCE_THRESHOLD = 20.0

        if prediction_summary['primary_prediction']['confidence'] < CONFIDENCE_THRESHOLD:
            prediction_summary['primary_prediction']['disease'] = 'Unknown'
            prediction_summary['confidence_level'] = 'Very Low'

        # Add treatment info
        for pred in prediction_summary['top_3']:
            pred['treatment'] = get_treatment_for_disease(pred['disease'])

        primary_disease = prediction_summary['primary_prediction']['disease']
        alternative_diseases = [pred['disease'] for pred in prediction_summary['top_3'][1:3]]

        # Save to database
        saved_image_id = None
        if not preview_only:
            try:
                image_file.seek(0)
                mango_image = MangoImage.objects.create(
                    image=image_file,
                    disease_classification=prediction_summary['primary_prediction']['disease'],
                    confidence_score=prediction_summary['primary_prediction']['confidence'],
                    detection_type=model_used,
                    ip_address=get_client_ip(request),
                    latitude=latitude,
                    longitude=longitude,
                    location_accuracy_confirmed=location_accuracy_confirmed,
                    location_source=location_source,
                    location_address=location_address,
                    selected_symptoms=selected_symptoms,
                    primary_symptoms=primary_symptoms,
                    alternative_symptoms=alternative_symptoms,
                    detected_disease=detected_disease,
                    is_detection_correct=is_detection_correct,
                    user_feedback=user_feedback,
                    plant_part_affected=plant_part_affected,
                    environmental_factors=environmental_factors
                )
                saved_image_id = mango_image.id
                
                try:
                    log_prediction_activity(
                        user=request.user if (hasattr(request, 'user') and request.user.is_authenticated) else None,
                        image=mango_image,
                        prediction_result=prediction_summary['primary_prediction']['disease'],
                        confidence=prediction_summary['primary_prediction']['confidence']
                    )
                except Exception as log_error:
                    print(f"Prediction activity logging error: {log_error}")
                    
            except Exception as db_error:
                print(f"Database save error: {db_error}")
                import traceback
                traceback.print_exc()

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
                'primary_disease_symptoms': [],
                'alternative_diseases': alternative_diseases
            },
            'user_verification': {
                'selected_symptoms': selected_symptoms,
                'primary_symptoms': primary_symptoms,
                'alternative_symptoms': alternative_symptoms,
                'detected_disease': detected_disease,
                'is_detection_correct': is_detection_correct,
                'user_feedback': user_feedback
            },
            'gate_validation': {
                'passed': True,
                'predicted_class': gate_prediction_label,
                'predicted_confidence': f"{gate_confidence:.2f}%" if gate_confidence else None,
                'mango_confidence': f"{mango_confidence:.2f}%" if mango_confidence else None,
                'message': f'Image validated as mango {detection_type}'
            },
            'model_used': model_used,
            'model_path': model_path
        }
        
        if not preview_only and saved_image_id:
            response_data['image_id'] = saved_image_id
            
        try:
            execution_time = time.time() - start_time
            response_data['execution_time'] = f"{execution_time:.2f}s"
        except:
            pass

        return JsonResponse(
            create_api_response(
                success=True,
                data=response_data,
                message='Image processed successfully'
            )
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
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
            'gate_confidence_threshold': {
                'leaf': GATE_CONFIDENCE_THRESHOLD_LEAF,
                'fruit': GATE_CONFIDENCE_THRESHOLD_FRUIT
            },
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