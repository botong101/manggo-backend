from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
from PIL import Image, ImageOps
import numpy as np
import os
import gc
import json
import time
from ..models import MangoImage, MLModel, PredictionLog, Notification
from .utils import (
    get_client_ip, validate_image_file, get_disease_type,
    calculate_confidence_level, get_prediction_summary,
    log_prediction_activity, 
    create_api_response
)


def get_tensorflow_runtime():
    """Load TensorFlow lazily so Django can start even if native TF runtime is unavailable."""
    try:
        import tensorflow as tf
        return tf, None
    except Exception as exc:
        return None, str(exc)



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
GATE_CONFIDENCE_THRESHOLD_LEAF = 60.0  # More lenient for diseased leaves
GATE_CONFIDENCE_THRESHOLD_FRUIT = 60.0
GATE_CONFIDENCE_THRESHOLD = 60.0  # Default threshold

# ==================== DISEASE MODEL CLASS NAMES ====================
# diseases the leaf model knows
LEAF_CLASS_NAMES = [
    'Anthracnose','Die Back', 'Healthy','Powdery Mildew','Sooty Mold',
]

FRUIT_CLASS_NAMES = [
    'Alternaria', 'Anthracnose', 'Black Mold Rot', 'Healthy', 'Stem end Rot'
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

    # 1. Query Disease table (any plant_part)
    try:
        from ..models import Disease as DiseaseModel
        norm = disease_name.replace('_', ' ').replace('-', ' ').strip().lower()
        for disease in DiseaseModel.objects.all():
            if disease.name.replace('_', ' ').replace('-', ' ').strip().lower() == norm:
                if disease.treatment:
                    return disease.treatment
                break  # found the record but it has no treatment yet — fall through
    except Exception:
        pass

    # 2. Fallback to hardcoded dict
    treatment = treatment_suggestions.get(disease_name)
    if treatment:
        return treatment
    disease_lower = disease_name.lower()
    for key, value in treatment_suggestions.items():
        if key.lower() == disease_lower:
            return value
    disease_normalized = disease_name.replace('_', ' ').replace('-', ' ').strip()
    for key, value in treatment_suggestions.items():
        if disease_normalized.lower() == key.replace('_', ' ').replace('-', ' ').strip().lower():
            return value

    return f"No treatment information available for '{disease_name}'. Please consult with an agricultural expert."

# fallback filenames if DB has no config yet
_DEFAULT_LEAF_MODEL       = 'mobilenetv2-leaf.keras'
_DEFAULT_FRUIT_MODEL      = 'mobilenetv2-fruit.keras'
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


def preprocess_image(image_file, target_size=None):
    """
    Preprocessing for inference — applies EXIF correction, resizes, returns
    float32 [0, 255].  The model's internal Rescaling layer handles normalization.

    target_size: (W, H) tuple; defaults to the global IMG_SIZE (224×224).
    """
    try:
        img = Image.open(image_file)
        img = ImageOps.exif_transpose(img)  # fix mobile camera rotation
        img = img.convert('RGB')
        original_size = img.size
        size = target_size if target_size else IMG_SIZE
        img = img.resize(size)
        img_array = np.array(img).astype("float32")  # [0, 255] float32
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, original_size
    except Exception as e:
        raise e


def _get_hybrid_specs(model):
    """
    For a dual-input MangoSenseNet-CoAttn model return (img_size, num_features).
    img_size is (W, H) matching PIL's resize convention.
    Detects by shape/rank so explicit input names are not required.
    """
    img_hw = None
    num_features = None
    for inp in model.inputs:
        s = inp.shape
        ndim = len(s)
        # Image branch: (batch, H, W, C) where C == 3
        if ndim == 4 and int(s[-1]) == 3:
            img_hw = (int(s[2]), int(s[1]))   # PIL resize wants (W, H)
        # Feature/symptom branch: (batch, N)
        elif ndim == 2:
            num_features = int(s[1])
    return img_hw, num_features


def _run_hybrid_model(model, image_file, img_size, num_features, symptom_vector=None):
    """
    Run hybrid model with image + symptom vector.
    If symptom_vector (shape (1, num_features)) is provided, uses it directly.
    Otherwise silences the symptom branch with zeros (image-only baseline).
    Returns (prediction_array, processed_size).
    Uses positional list input to avoid relying on Keras auto-generated input names.
    """
    img_array, _ = preprocess_image(image_file, target_size=img_size)
    sym = symptom_vector if symptom_vector is not None \
          else np.zeros((1, num_features), dtype=np.float32)
    # Positional list: model.inputs[0]=image, model.inputs[1]=symptoms
    pred = model.predict([img_array, sym])
    return np.array(pred).flatten(), img_size



@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([IsAuthenticated])
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
        tf, tf_runtime_error = get_tensorflow_runtime()
        if tf is None:
            return JsonResponse(
                create_api_response(
                    success=False,
                    message='TensorFlow runtime unavailable',
                    errors=[tf_runtime_error]
                ),
                status=503
            )

        image_file = request.FILES['image']
        
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
                    threshold = GATE_CONFIDENCE_THRESHOLD_FRUIT
                else:
                    valid_idx = GATE_VALID_INDEX_LEAF
                    gate_cls = GATE_LEAF_CLASS_NAMES
                    threshold = GATE_CONFIDENCE_THRESHOLD_LEAF

                # Get the predicted class and its confidence
                gate_predicted_idx = int(np.argmax(gate_pred))
                gate_prediction_label = gate_cls[gate_predicted_idx]
                gate_predicted_confidence = float(gate_pred[gate_predicted_idx]) * 100
                
                # Get mango confidence specifically
                mango_confidence = float(gate_pred[valid_idx]) * 100
                gate_confidence = mango_confidence  # For response
                
                # STRICT VALIDATION:
                # 1. The highest predicted class MUST be "Mango"
                # 2. AND the mango confidence must be above threshold
                is_mango_predicted = (gate_predicted_idx == valid_idx)
                is_confidence_sufficient = (mango_confidence >= threshold)
                
                gate_passed = is_mango_predicted and is_confidence_sufficient
                
                # Debug logging
                print(f"=== GATE VALIDATION DEBUG ===")
                print(f"Detection type: {detection_type}")
                print(f"Gate predictions: {dict(zip(gate_cls, [f'{p*100:.2f}%' for p in gate_pred]))}")
                print(f"Predicted class: {gate_prediction_label} ({gate_predicted_confidence:.2f}%)")
                print(f"Mango confidence: {mango_confidence:.2f}%")
                print(f"Threshold: {threshold}%")
                print(f"Is Mango predicted: {is_mango_predicted}")
                print(f"Is confidence sufficient: {is_confidence_sufficient}")
                print(f"GATE PASSED: {gate_passed}")
                print(f"=============================")

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
            
            # Build a more informative rejection message
            if gate_prediction_label and gate_prediction_label.lower() != 'mango':
                rejection_detail = f'The image appears to be a {gate_prediction_label} {detection_type}, not a Mango {detection_type}.'
            else:
                rejection_detail = f'The image does not appear to be a clear Mango {detection_type}. Mango confidence: {gate_confidence:.1f}%'
            
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
                                f"{rejection_detail} "
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
                            'gate_predicted_confidence': gate_predicted_confidence if 'gate_predicted_confidence' in locals() else None,
                            'mango_confidence': gate_confidence,
                            'threshold': threshold if 'threshold' in locals() else GATE_CONFIDENCE_THRESHOLD,
                            'message': rejection_detail
                        },
                        'saved_image_id': None,
                        'model_used': detection_type,
                        'debug_info': {
                            'gate_model_used': True,
                            'gate_model_path': gate_model_path,
                            'processing_time': time.time() - start_time,
                            'image_size': original_size,
                            'inference_mode': 'gate_rejected'
                        }
                    },
                    message=rejection_detail
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
        except Exception as first_load_error:
            # Keras version mismatch: newer models may have quantization_config in Dense
            # which older Keras versions don't recognise. Patch Dense.__init__ during
            # load so the unknown kwarg is silently dropped, then restore immediately.
            if 'quantization_config' in str(first_load_error):
                _orig_dense_init = tf.keras.layers.Dense.__init__
                def _compat_dense_init(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    _orig_dense_init(self, *args, **kwargs)
                tf.keras.layers.Dense.__init__ = _compat_dense_init
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
                finally:
                    tf.keras.layers.Dense.__init__ = _orig_dense_init
            else:
                return JsonResponse(
                    create_api_response(
                        success=False,
                        message='Failed to load ML model',
                        errors=[str(first_load_error)]
                    ),
                    status=500
                )

        # run prediction — handle dual-input hybrid model separately
        try:
            is_hybrid = len(model.inputs) > 1

            if is_hybrid:
                # MangoSenseNet-CoAttn: two-pass self-confirming inference
                img_size, num_features = _get_hybrid_specs(model)
                if img_size is None:
                    img_size = (240, 240)
                if num_features is None:
                    num_features = int(model.inputs[1].shape[1])

                # Load class prototypes from sidecar if present
                model_stem = os.path.splitext(os.path.basename(model_path))[0]
                proto_path = os.path.join(os.path.dirname(model_path),
                                          f"{model_stem}_prototypes.json")
                prototypes = {}
                if os.path.exists(proto_path):
                    with open(proto_path) as _f:
                        prototypes = json.load(_f)  # {class_name: [float, ...]}
                    model_class_names = list(prototypes.keys())
                # else model_class_names already set from detection_type above

                # Pass 1 — zero symptoms: image branch makes the initial call
                image_file.seek(0)
                prediction_p1, processed_size = _run_hybrid_model(
                    model, image_file, img_size, num_features
                )

                # Determine which prototype to use for Pass 2
                inference_mode = "zero_symptoms"
                symptom_vector = None

                if prototypes:
                    # User-confirmed disease takes priority; otherwise trust Pass 1
                    if detected_disease and detected_disease in prototypes:
                        target_class = detected_disease
                        inference_mode = "user_confirmed_prototype"
                    else:
                        target_class = model_class_names[int(np.argmax(prediction_p1))]
                        inference_mode = "self_confirmed_prototype"

                    proto_values = prototypes.get(target_class)
                    if proto_values is not None:
                        symptom_vector = np.array([proto_values], dtype=np.float32)

                # Pass 2 — prototype symptoms: refined prediction
                if symptom_vector is not None:
                    image_file.seek(0)
                    prediction, processed_size = _run_hybrid_model(
                        model, image_file, img_size, num_features, symptom_vector
                    )
                else:
                    prediction = prediction_p1  # no sidecar → use Pass 1 as-is
            else:
                # Standard single-input model (MobileNetV2 etc.)
                prediction = model.predict(img_array)
                prediction = np.array(prediction).flatten()
                processed_size = IMG_SIZE
                inference_mode = "standard"

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
                    'processed_size': processed_size,
                    'inference_mode': inference_mode
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
                
                # PIL.Image.open() consumed the file pointer during preprocessing.
                # Reset to start so S3Boto3Storage uploads the full file, not 0 bytes.
                image_file.seek(0)

                # Give the file a unique temp name before S3 upload.
                # The mobile app always sends "image.jpg" which would overwrite
                # the previous file in the bucket on every detection.
                import uuid as _uuid
                _ext = os.path.splitext(image_file.name)[-1].lower() or '.jpg'
                image_file.name = f"tmp_{_uuid.uuid4().hex}{_ext}"

                mango_image = MangoImage.objects.create(
                    image=image_file,
                    original_filename=image_file.name,
                    predicted_class=prediction_summary['primary_prediction']['disease'],
                    disease_classification=prediction_summary['primary_prediction']['disease'],
                    disease_type=model_used, 
                    model_used=model_used,  # Store which model was actually used
                    model_filename=os.path.basename(model_path),  # Store the actual model filename
                    confidence_score=prediction_summary['primary_prediction']['confidence'] / 100,
                    user=request.user if request.user.is_authenticated else None,
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
                log_prediction_activity(request.user, mango_image.id, prediction_summary)
                saved_image_id = mango_image.id

                # Rename S3 file to {id}_{disease}_{confidence}pct.ext
                # S3 has no native rename — copy to new key then delete old key.
                try:
                    import boto3 as _boto3
                    disease_slug = mango_image.predicted_class.replace(' ', '_')
                    conf_pct = int(mango_image.confidence_score * 100)
                    _ext_final = os.path.splitext(mango_image.image.name)[-1] or '.jpg'
                    old_key = mango_image.image.name
                    new_key = f"mango_images/{mango_image.id}_{disease_slug}_{conf_pct}pct{_ext_final}"

                    _s3 = _boto3.client(
                        's3',
                        endpoint_url=settings.AWS_S3_ENDPOINT_URL,
                        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                        region_name=settings.AWS_S3_REGION_NAME,
                    )
                    _s3.copy_object(
                        Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                        CopySource={'Bucket': settings.AWS_STORAGE_BUCKET_NAME, 'Key': old_key},
                        Key=new_key,
                    )
                    _s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=old_key)

                    mango_image.image.name = new_key
                    mango_image.save(update_fields=['image'])
                except Exception as _rename_err:
                    print(f"[RENAME WARNING] S3 rename failed, keeping temp name: {_rename_err}")

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
                import traceback
                print(f"[S3 UPLOAD ERROR] {type(e).__name__}: {e}")
                print(traceback.format_exc())
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
                'processed_size': processed_size,
                'inference_mode': inference_mode
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