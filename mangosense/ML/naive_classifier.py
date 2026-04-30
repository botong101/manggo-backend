"""
Pure-Python Bernoulli Naive Bayes classifier for mango symptom -> disease.

NO external dependencies. Everything is stdlib (math, json, collections, pathlib,
time, threading, typing). Trained model is a single JSON file — hot-reloadable,
inspectable, version-control friendly.
"""

import math
import json
import time
import threading
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable

from mangosense.repositories.symptom_repository import (
    get_vocabulary, get_diseases, normalize_symptom,
)
from mangosense.models import Symptom as SymptomModel


# ── A. Feature encoder ────────────────────────────────────────────────────────

class SymptomEncoder:
    """Convert a list of raw symptom strings into a binary feature vector.

    Uses DB vector_index as the authoritative slot number — not enumerate
    position — so gaps from deleted rows cannot silently shift columns.
    """

    def __init__(self, plant_part: str):
        vocabulary_symptom_rows = (
            SymptomModel.objects
            .filter(plant_part=plant_part, is_in_vocabulary=True)
            .exclude(vector_index=None)
            .values('key', 'vector_index')
            .order_by('vector_index')
        )
        self.symptom_to_slot: dict[str, int] = {
            symptom_row['key']: symptom_row['vector_index']
            for symptom_row in vocabulary_symptom_rows
        }
        self.feature_vector_length: int = (
            max(self.symptom_to_slot.values()) + 1 if self.symptom_to_slot else 0
        )
        self.vocabulary: list[str] = get_vocabulary(plant_part)

    def encode(self, raw_symptoms: Iterable[str]) -> list[int]:
        """Return [0/1, ...] of length self.feature_vector_length."""
        feature_vector = [0] * self.feature_vector_length
        for raw_symptom_text in raw_symptoms or ():
            canonical_symptom_key = normalize_symptom(raw_symptom_text)
            feature_slot_number = self.symptom_to_slot.get(canonical_symptom_key)
            if feature_slot_number is not None:
                feature_vector[feature_slot_number] = 1
        return feature_vector


# ── B. Classifier ─────────────────────────────────────────────────────────────

class NaiveBayesSymptomClassifier:
    """Bernoulli NB with Laplace smoothing, IDF evidence weights, log-sum-exp,
    and temperature softening.

    Public API:
        fit(records)             — train from (symptoms, disease) pairs
        predict_proba(symptoms)  — {disease: probability_0_to_1}
        predict(symptoms)        — (top_disease | 'Uncertain', confidence, dist)
        save(path)               — write JSON model file
        load(path)               — class method: load JSON model file
        feature_importances()    — sorted list of {symptom, weight}
    """

    def __init__(
        self,
        plant_part: str,
        alpha: float = 1.0,
        temperature: float = 1.5,
        min_confidence: float = 0.40,
        min_margin: float = 0.10,
        evidence_weight_floor: float = 0.05,
    ):
        self.plant_part = plant_part
        self.alpha = alpha
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.evidence_weight_floor = evidence_weight_floor

        self.encoder = SymptomEncoder(plant_part)
        self.vocabulary: list[str] = self.encoder.vocabulary
        self.feature_vector_length: int = self.encoder.feature_vector_length
        self.disease_classes: list[str] = []
        self.log_priors_per_disease: dict[str, float] = {}
        self.log_likelihoods_per_disease: dict[str, list[tuple[float, float]]] = {}
        self.symptom_evidence_weights: list[float] = []
        self.metadata: dict = {}

    # ── B.1 Training ──────────────────────────────────────────────────────────

    def fit(self, training_records: list[tuple[list[str], str]]) -> None:
        """Train on (raw_symptom_list, disease_name) pairs."""
        sample_count_per_disease: Counter = Counter()
        presence_counts_per_disease: dict[str, list[int]] = defaultdict(
            lambda: [0] * self.feature_vector_length
        )

        for raw_symptom_list, disease_name in training_records:
            sample_count_per_disease[disease_name] += 1
            feature_vector = self.encoder.encode(raw_symptom_list)
            for feature_slot_number, slot_value in enumerate(feature_vector):
                if slot_value == 1:
                    presence_counts_per_disease[disease_name][feature_slot_number] += 1

        self.disease_classes = sorted(sample_count_per_disease.keys())
        total_sample_count = sum(sample_count_per_disease.values())
        disease_class_count = len(self.disease_classes)

        self.log_priors_per_disease = {
            disease_name: math.log(
                (sample_count_per_disease[disease_name] + self.alpha)
                / (total_sample_count + self.alpha * disease_class_count)
            )
            for disease_name in self.disease_classes
        }

        self.log_likelihoods_per_disease = {}
        for disease_name in self.disease_classes:
            disease_sample_count = sample_count_per_disease[disease_name]
            smoothed_denominator = disease_sample_count + 2 * self.alpha
            log_likelihood_row: list[tuple[float, float]] = []
            for feature_slot_number in range(self.feature_vector_length):
                presence_count = presence_counts_per_disease[disease_name][feature_slot_number]
                conditional_probability = (presence_count + self.alpha) / smoothed_denominator
                conditional_probability = min(max(conditional_probability, 1e-9), 1 - 1e-9)
                log_likelihood_row.append((
                    math.log(conditional_probability),
                    math.log(1.0 - conditional_probability),
                ))
            self.log_likelihoods_per_disease[disease_name] = log_likelihood_row

        self.symptom_evidence_weights = []
        for feature_slot_number in range(self.feature_vector_length):
            disease_frequency_for_slot = sum(
                1 for disease_name in self.disease_classes
                if presence_counts_per_disease[disease_name][feature_slot_number] > 0
            )
            if disease_frequency_for_slot == 0:
                evidence_weight = self.evidence_weight_floor
            else:
                evidence_weight = max(
                    math.log(disease_class_count / disease_frequency_for_slot),
                    self.evidence_weight_floor,
                )
            self.symptom_evidence_weights.append(evidence_weight)

        self.metadata = {
            'total_sample_count':       total_sample_count,
            'disease_class_count':      disease_class_count,
            'sample_count_per_disease': dict(sample_count_per_disease),
            'training_date':            time.strftime('%Y-%m-%dT%H:%M:%S'),
            'alpha':                    self.alpha,
            'temperature':              self.temperature,
        }

    # ── B.2 Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, raw_symptoms: list[str]) -> dict[str, float]:
        """Return {disease_name: probability_0_to_1}, sorted descending."""
        if not self.disease_classes:
            return {}

        feature_vector = self.encoder.encode(raw_symptoms)

        log_posteriors_per_disease: dict[str, float] = {}
        for disease_name in self.disease_classes:
            log_score = self.log_priors_per_disease[disease_name]
            log_likelihood_row = self.log_likelihoods_per_disease[disease_name]
            for feature_slot_number, slot_value in enumerate(feature_vector):
                log_likelihood_for_slot = (
                    log_likelihood_row[feature_slot_number][0] if slot_value == 1
                    else log_likelihood_row[feature_slot_number][1]
                )
                log_score += (
                    self.symptom_evidence_weights[feature_slot_number]
                    * log_likelihood_for_slot
                )
            log_posteriors_per_disease[disease_name] = log_score

        temperature_value = max(self.temperature, 1e-6)
        scaled_log_posteriors = {
            disease_name: log_posteriors_per_disease[disease_name] / temperature_value
            for disease_name in self.disease_classes
        }
        max_scaled_log = max(scaled_log_posteriors.values())
        unnormalised_exp_per_disease = {
            disease_name: math.exp(scaled_log_posteriors[disease_name] - max_scaled_log)
            for disease_name in self.disease_classes
        }
        normalisation_constant = sum(unnormalised_exp_per_disease.values())
        probabilities_per_disease = {
            disease_name: unnormalised_exp_per_disease[disease_name] / normalisation_constant
            for disease_name in self.disease_classes
        }
        return dict(sorted(
            probabilities_per_disease.items(),
            key=lambda disease_probability_pair: disease_probability_pair[1],
            reverse=True,
        ))

    def predict(self, raw_symptoms: list[str]) -> tuple[str, float, dict[str, float]]:
        """Return (top_disease_or_Uncertain, confidence_0_to_1, full_distribution)."""
        probabilities_per_disease = self.predict_proba(raw_symptoms)
        if not probabilities_per_disease:
            return ('Uncertain', 0.0, {})

        ordered_disease_probabilities = list(probabilities_per_disease.items())
        top_disease_name, top_probability = ordered_disease_probabilities[0]
        second_probability = (
            ordered_disease_probabilities[1][1]
            if len(ordered_disease_probabilities) > 1 else 0.0
        )
        confidence_margin = top_probability - second_probability

        if top_probability < self.min_confidence or confidence_margin < self.min_margin:
            return ('Uncertain', top_probability, probabilities_per_disease)
        return (top_disease_name, top_probability, probabilities_per_disease)

    # ── B.3 Persistence ───────────────────────────────────────────────────────

    def save(self, model_file_path: str | Path) -> None:
        """Write the trained model to a single JSON file."""
        serialised_model_payload = {
            'format_version':              2,
            'plant_part':                  self.plant_part,
            'disease_classes':             self.disease_classes,
            'vocabulary':                  self.vocabulary,
            'symptom_to_slot':             self.encoder.symptom_to_slot,
            'feature_vector_length':       self.feature_vector_length,
            'log_priors_per_disease':      self.log_priors_per_disease,
            'log_likelihoods_per_disease': {
                disease_name: [
                    [log_present, log_absent]
                    for (log_present, log_absent) in log_likelihood_row
                ]
                for disease_name, log_likelihood_row in self.log_likelihoods_per_disease.items()
            },
            'symptom_evidence_weights':    self.symptom_evidence_weights,
            'hyperparameters': {
                'alpha':                 self.alpha,
                'temperature':           self.temperature,
                'min_confidence':        self.min_confidence,
                'min_margin':            self.min_margin,
                'evidence_weight_floor': self.evidence_weight_floor,
            },
            'metadata': self.metadata,
        }
        model_file_path = Path(model_file_path)
        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        model_file_path.write_text(json.dumps(serialised_model_payload, indent=2))

    @classmethod
    def load(cls, model_file_path: str | Path) -> 'NaiveBayesSymptomClassifier':
        """Load a previously-saved model. Returns a ready-to-predict instance."""
        serialised_model_payload = json.loads(Path(model_file_path).read_text())
        hyperparameters = serialised_model_payload.get('hyperparameters', {})
        loaded_instance = cls(
            plant_part=serialised_model_payload['plant_part'],
            alpha=hyperparameters.get('alpha', 1.0),
            temperature=hyperparameters.get('temperature', 1.5),
            min_confidence=hyperparameters.get('min_confidence', 0.40),
            min_margin=hyperparameters.get('min_margin', 0.10),
            evidence_weight_floor=hyperparameters.get('evidence_weight_floor', 0.05),
        )
        loaded_instance.disease_classes = serialised_model_payload['disease_classes']
        loaded_instance.vocabulary = serialised_model_payload['vocabulary']
        loaded_instance.encoder.vocabulary = serialised_model_payload['vocabulary']

        saved_symptom_to_slot = serialised_model_payload.get('symptom_to_slot')
        if saved_symptom_to_slot:
            loaded_instance.encoder.symptom_to_slot = {
                symptom_key: slot_number
                for symptom_key, slot_number in saved_symptom_to_slot.items()
            }
        else:
            loaded_instance.encoder.symptom_to_slot = {
                symptom_key: slot_position
                for slot_position, symptom_key in enumerate(serialised_model_payload['vocabulary'])
            }
        loaded_instance.encoder.feature_vector_length = (
            max(loaded_instance.encoder.symptom_to_slot.values()) + 1
            if loaded_instance.encoder.symptom_to_slot else 0
        )
        loaded_instance.feature_vector_length = loaded_instance.encoder.feature_vector_length
        loaded_instance.log_priors_per_disease = serialised_model_payload['log_priors_per_disease']
        loaded_instance.log_likelihoods_per_disease = {
            disease_name: [
                (log_likelihood_pair[0], log_likelihood_pair[1])
                for log_likelihood_pair in log_likelihood_rows
            ]
            for disease_name, log_likelihood_rows
            in serialised_model_payload['log_likelihoods_per_disease'].items()
        }
        loaded_instance.symptom_evidence_weights = serialised_model_payload['symptom_evidence_weights']
        loaded_instance.metadata = serialised_model_payload.get('metadata', {})
        return loaded_instance

    # ── B.4 Introspection ─────────────────────────────────────────────────────

    def feature_importances(self) -> list[dict]:
        """Return symptoms ranked by evidence weight, descending."""
        ranked_symptom_weights = sorted(
            [
                {
                    'symptom': symptom_key,
                    'weight':  round(self.symptom_evidence_weights[slot_number], 4),
                }
                for symptom_key, slot_number in self.encoder.symptom_to_slot.items()
            ],
            key=lambda symptom_weight_entry: symptom_weight_entry['weight'],
            reverse=True,
        )
        return ranked_symptom_weights


# ── C. Module-level cache + thin functional API ───────────────────────────────

_MODEL_CACHE: dict[str, 'NaiveBayesSymptomClassifier | None'] = {}
_CACHE_LOCK = threading.Lock()


def _model_path(plant_part: str) -> Path:
    """Resolve the on-disk model file path for a given plant_part."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / 'models' / f'naive_classifier_{plant_part}.json'


def load_naive_model(plant_part: str) -> 'NaiveBayesSymptomClassifier | None':
    """Return cached model or None if file missing. Thread-safe."""
    with _CACHE_LOCK:
        if plant_part in _MODEL_CACHE:
            return _MODEL_CACHE[plant_part]

        path = _model_path(plant_part)
        if not path.exists():
            _MODEL_CACHE[plant_part] = None
            return None

        try:
            instance = NaiveBayesSymptomClassifier.load(path)
            _MODEL_CACHE[plant_part] = instance
            return instance
        except Exception as error:
            print(f'[NAIVE CLASSIFIER] Failed to load {path}: {error}')
            _MODEL_CACHE[plant_part] = None
            return None


def invalidate_naive_cache(plant_part: str | None = None) -> None:
    """Bust the in-memory cache after training writes a new model file."""
    with _CACHE_LOCK:
        if plant_part is None:
            _MODEL_CACHE.clear()
        else:
            _MODEL_CACHE.pop(plant_part, None)


def predict_from_symptoms_naive(raw_symptoms: list[str], plant_part: str) -> dict | None:
    """Public entry point for ml_views.py — returns same dict shape as Plan 04."""
    if not raw_symptoms:
        return None
    model = load_naive_model(plant_part)
    if model is None:
        return None

    top_disease, top_prob, probabilities = model.predict(raw_symptoms)

    probabilities_pct = {disease: round(prob * 100, 2) for disease, prob in probabilities.items()}
    confidence_pct = round(top_prob * 100, 2)

    from mangosense.views.utils import calculate_confidence_level
    confidence_level = calculate_confidence_level(confidence_pct)

    return {
        'probabilities':    probabilities_pct,
        'top_prediction':   top_disease,
        'confidence':       confidence_pct,
        'confidence_level': confidence_level,
        'classifier':       'naive_bayes_v1',
    }


# ── D. Ensemble fusion ────────────────────────────────────────────────────────

def fuse_image_and_symptom_naive(
    image_probs: dict[str, float],
    symptom_probs: dict[str, float],
    disease_names: list[str],
) -> dict:
    """Agreement-based weighted fusion of CNN and NB probabilities (0–100 scale)."""
    image_array   = [image_probs.get(disease, 0.0) for disease in disease_names]
    symptom_array = [symptom_probs.get(disease, 0.0) for disease in disease_names]

    top_image_disease   = disease_names[max(range(len(image_array)),   key=lambda slot_index: image_array[slot_index])]
    top_symptom_disease = disease_names[max(range(len(symptom_array)), key=lambda slot_index: symptom_array[slot_index])]
    image_confidence    = max(image_array)
    symptom_confidence  = max(symptom_array)

    if top_image_disease == top_symptom_disease:
        fused = [0.7 * image_array[slot_index] + 0.3 * symptom_array[slot_index] for slot_index in range(len(disease_names))]
        reasoning = f'Image and symptoms agree on {top_image_disease}'
    elif image_confidence > 75.0:
        fused = list(image_array)
        reasoning = (
            f'Image highly confident ({image_confidence:.1f}%), '
            f'overriding symptom suggestion ({top_symptom_disease})'
        )
    elif symptom_confidence > 75.0:
        fused = list(symptom_array)
        reasoning = (
            f'Symptoms highly confident ({symptom_confidence:.1f}%), '
            f'overriding image suggestion ({top_image_disease})'
        )
    else:
        fused = [0.6 * image_array[slot_index] + 0.4 * symptom_array[slot_index] for slot_index in range(len(disease_names))]
        reasoning = (
            f'Both uncertain (image: {top_image_disease} {image_confidence:.1f}%, '
            f'symptoms: {top_symptom_disease} {symptom_confidence:.1f}%) — expert review recommended'
        )

    total_fused_score = sum(fused)
    if total_fused_score > 0:
        fused = [score / total_fused_score * 100.0 for score in fused]

    probabilities_by_disease = {disease_names[slot_index]: round(fused[slot_index], 2) for slot_index in range(len(disease_names))}
    top_index = max(range(len(fused)), key=lambda slot_index: fused[slot_index])

    from mangosense.views.utils import calculate_confidence_level
    confidence = round(fused[top_index], 2)
    return {
        'probabilities':    probabilities_by_disease,
        'top_prediction':   disease_names[top_index],
        'confidence':       confidence,
        'confidence_level': calculate_confidence_level(confidence),
        'reasoning':        reasoning,
    }
