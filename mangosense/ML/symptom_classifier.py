import os
import json
import threading
import datetime
import numpy as np
from django.conf import settings

from .symptom_vocabulary import(
    normalize_symptom, get_vocabulary, get_diseases,
    LEAF_SYMPTOMS, FRUIT_SYMPTOMS
)
#Featyre encoder
class SymptomFeatureEncoder:
    """Converts a list of symptom strings into a binary feature vector."""

    def __init__(self, disease_type: str):
        self.vocabulary = get_vocabulary(disease_type)
        self._symptom_to_index = {symptom: position for position, 
                                  symptom in enumerate(self.vocabulary)}
        
    def encode(slef, symptoms: list) -> np.ndarray:
        """Return a binary vector of shape (len(vocabulary),)."""
        feature_vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        for symptom in symptoms:
            canonical = normalize_symptom(symptom)
            if canonical in self._symptom_to_index:
                feature_vector[self._symptom_to_index[canonical]] = 1.0
        return feature_vector

    def decode_vector(self, vector: np.ndarray) -> list:
        """Inverse of encode — returns list of active symptom names (for debugging).""" 
        return [self.vocabulary[position] for position, value in enumerate(vector) if value > 0]
    
#Inference
_XGBOOST_MODELS: dict = {}
_model_lock = threading.Lock()

def _model_path(disease_type: str) -> str:
    base = getattr(settings, 'BASE_DIR', 
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(__file__)
                        )
                    )
                   )
    return os.path.join(base, 'models', f'symptom_classifier_{disease_type}.json')

def _meta_path(disease_type: str) -> str:
    base = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(base, 'models', f'symptom_classifier_{disease_type}_meta.json')

def load_symptom_model(disease_type: str):
    """
    Load (or return cached) XGBoost model + meta for the given disease_type.
    Returns (model, meta, encoder) or (None, None, None) if not trained yet.
    """
    with _model_lock:
        if disease_type in _XGBOOST_MODELS:
            return _XGBOOST_MODELS[disease_type]
        
        model_file_path = _model_path(disease_type)
        meta_file_path = _meta_path(disease_type)

        if not os.path.exists(model_file_path):
            _XGBOOST_MODELS[disease_type] = (None, None, None)
            return (None, None, None)

        try:
            from xgboost import XGBClassifier
            model = XGBClassifier

            model.load_model(model_file_path)

            meta = {}
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'r') as meta_file:
                    meta = json.load(meta_file)
            
            encoder = SymptomFeatureEncoder(disease_type)
            _XGBOOST_MODELS[disease_type] = (model, meta, encoder)
            return (model, meta, encoder)
        
        except Exception as error:
            print(f"[SYMPTOM CLASSIFIER] Failed to load model for {disease_type}: {error}")

            #cache the failure so we dont try to load a broken file on every request.
            _XGBOOST_MODELS[disease_type] = (None, None, None)
            return (None, None, None)

def invalidate_model_cache(disease_type: str):
    """Force reload on next call — call this after training completes."""
    with _model_lock:
        _XGBOOST_MODELS.pop(disease_type, None)


def predict_from_symptoms(symptoms: list, disease_type: str) -> dict | None:
    """
    Run XGBoost inference on a symptom list.

    Returns:
        {
            'probabilities': {'Anthracnose': 0.82, 'Die Back': 0.10, ...},
            'top_prediction': 'Anthracnose',
            'confidence': 82.0,   # percent
            'confidence_level': 'High',
        }
        or None if the model is not trained yet or symptoms list is empty.
    """

    if not symptoms:
        return None
    
    model, meta, encoder = load_symptom_model(disease_type)

    if model is None:
        return None
    
    diseases = meta.get('classes', get_diseases(disease_type))

    feature_vector = encoder.encode(symptoms)

    probabilities_array = model.predict_proba(feature_vector.reshape(1,-1))[0]


    probabilities_by_disease = {
        diseases[class_index]: round(float(probabilities_array[class_index]) * 100, 2)

        for class_index in range(len(diseases))
    }

    top_class_index = int(np.argmax(probabilities_array))
    top_disease = diseases[top_class_index]
    confidence = round(float(probabilities_array[top_class_index]) * 100, 2)

    #fix sa fuckin circular import :) 
    from .symptom_vocabulary import normalize_symptom
    from mangosense.views.utils import calculate_confidence_level
    confidence_level = calculate_confidence_level(confidence)

    return{
        'probabilities': probabilities_by_disease,
        'top_prediction': top_disease,
        'confidence': confidence,
        'confidence_level': confidence_level 
    }


#fuseion of image and symptoms
def fuse_image_and_symptoms(
        image_probs: dict,
        symptom_probs: dict,
        disease_names: list,
) -> dict:
    """
    Args:
        image_probs:   {disease: probability_0_to_100}  from CNN
        symptom_probs: {disease: probability_0_to_100}  from XGBoost
        disease_names: ordered list of disease class names

    Returns:
        {
            'probabilities': {disease: fused_score},
            'top_prediction': str,
            'confidence': float,
            'confidence_level': str,
            'reasoning': str,
        }
    """
    image_probabilities_array = np.array(
        [image_probs.get(disease, 0.0) for disease in disease_names],
        dtype = np.float32
    )
    symptom_probabilities_array = np.array(
        [image_probs.get(disease, 0.0) for disease in disease_names],
        dtype = np.float32
    )

    top_image_disease = disease_names[int(np.argmax(image_probabilities_array))]
    top_symptom_disease = disease_names[int(np.argmax(symptom_probabilities_array))]
    image_confidence = float(np.max(image_probabilities_array))
    symptom_confidence = float(np.max(symptom_probabilities_array))

    if top_image_disease == top_symptom_disease:
        

    


