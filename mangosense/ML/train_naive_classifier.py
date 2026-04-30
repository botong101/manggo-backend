"""
Pure-Python trainer for NaiveBayesSymptomClassifier.

Reads verified MangoImage rows from the DB, splits train/test in pure Python,
fits the classifier, evaluates accuracy + per-class precision/recall +
confusion matrix, and saves the JSON model file.

Usage:
    python manage.py shell -c "from mangosense.ML.train_naive_classifier import train_one; train_one('leaf')"
    python manage.py shell -c "from mangosense.ML.train_naive_classifier import train_one; train_one('fruit')"
"""

import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .naive_classifier import (
    NaiveBayesSymptomClassifier,
    _model_path,
    invalidate_naive_cache,
)

MIN_SAMPLES_PER_CLASS = 5
TEST_FRACTION = 0.2
SEED = 42


# ── A. Data collection ────────────────────────────────────────────────────────

def _fetch_records(plant_part: str) -> list[tuple[list[str], str]]:
    """Return [(raw_symptoms_list, disease_name), ...] from the DB."""
    from mangosense.models import MangoImage

    queryset = (
        MangoImage.objects
        .filter(training_ready=True, disease_type=plant_part)
        .filter(selected_symptoms__isnull=False)
        .exclude(selected_symptoms=[])
        .exclude(disease_classification='')
        .exclude(disease_classification__isnull=True)
        .values('selected_symptoms', 'disease_classification')
    )

    records: list[tuple[list[str], str]] = []
    for row in queryset:
        symptoms = row['selected_symptoms'] if isinstance(row['selected_symptoms'], list) else []
        disease = (row['disease_classification'] or '').strip()
        if disease and symptoms:
            records.append((symptoms, disease))
    return records


# ── B. Pure-Python train/test split ───────────────────────────────────────────

def train_test_split_py(
    records: list[tuple[list[str], str]],
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[list, list]:
    """Stratified split: each class bucket is shuffled and split independently."""
    rng = random.Random(seed)
    by_class: dict[str, list] = defaultdict(list)
    for record in records:
        by_class[record[1]].append(record)

    train_split: list = []
    test_split: list = []
    for disease_name, disease_bucket in by_class.items():
        rng.shuffle(disease_bucket)
        test_size = max(1, int(len(disease_bucket) * test_fraction))
        test_split.extend(disease_bucket[:test_size])
        train_split.extend(disease_bucket[test_size:])

    rng.shuffle(train_split)
    rng.shuffle(test_split)
    return train_split, test_split


# ── C. Pure-Python evaluation ─────────────────────────────────────────────────

def evaluate(
    model: NaiveBayesSymptomClassifier,
    test_records: list[tuple[list[str], str]],
) -> dict:
    """Return {accuracy, per_class, confusion_matrix, n_test_samples}."""
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct_count = 0
    total_count = 0

    for raw_symptoms, true_disease_name in test_records:
        predicted_disease_name, _confidence, _probs = model.predict(raw_symptoms)
        confusion[true_disease_name][predicted_disease_name] += 1
        if predicted_disease_name == true_disease_name:
            correct_count += 1
        total_count += 1

    overall_accuracy = correct_count / total_count if total_count else 0.0

    per_class: dict[str, dict[str, float]] = {}
    all_class_labels = sorted(
        set(list(confusion.keys()) + [
            predicted_label
            for row_dict in confusion.values()
            for predicted_label in row_dict
        ])
    )
    for evaluated_class in all_class_labels:
        true_positive_count = confusion.get(evaluated_class, {}).get(evaluated_class, 0)
        false_negative_count = sum(
            count for predicted_label, count in confusion.get(evaluated_class, {}).items()
            if predicted_label != evaluated_class
        )
        false_positive_count = sum(
            confusion.get(other_class, {}).get(evaluated_class, 0)
            for other_class in all_class_labels if other_class != evaluated_class
        )
        precision_score = (
            true_positive_count / (true_positive_count + false_positive_count)
            if (true_positive_count + false_positive_count) else 0.0
        )
        recall_score = (
            true_positive_count / (true_positive_count + false_negative_count)
            if (true_positive_count + false_negative_count) else 0.0
        )
        per_class[evaluated_class] = {
            'precision': round(precision_score, 4),
            'recall':    round(recall_score, 4),
            'support':   sum(confusion.get(evaluated_class, {}).values()),
        }

    return {
        'accuracy':         round(overall_accuracy, 4),
        'per_class':        per_class,
        'confusion_matrix': {true_label: dict(row_dict) for true_label, row_dict in confusion.items()},
        'n_test_samples':   total_count,
    }


# ── D. Top-level training entry point ─────────────────────────────────────────

def train_one(plant_part: str) -> dict:
    """Train, evaluate, save. Returns the metrics dict."""
    started_at = time.time()
    print(f'[train_naive_classifier] Fetching records for plant_part={plant_part}...')
    all_records = _fetch_records(plant_part)

    disease_sample_counts = Counter(disease_name for _symptoms, disease_name in all_records)
    eligible_disease_set = {
        disease_name for disease_name, sample_count in disease_sample_counts.items()
        if sample_count >= MIN_SAMPLES_PER_CLASS
    }
    if len(eligible_disease_set) < 2:
        return {
            'success': False,
            'error':   (
                f'Need >=2 classes with >={MIN_SAMPLES_PER_CLASS} samples each. '
                f'Counts: {dict(disease_sample_counts)}'
            ),
        }

    eligible_records = [
        (symptoms, disease_name) for symptoms, disease_name in all_records
        if disease_name in eligible_disease_set
    ]

    print(f'[train_naive_classifier] {len(eligible_records)} records across {len(eligible_disease_set)} classes.')
    train_records, test_records = train_test_split_py(eligible_records)
    print(f'[train_naive_classifier] split: {len(train_records)} train / {len(test_records)} test')

    trained_model = NaiveBayesSymptomClassifier(plant_part=plant_part)
    trained_model.fit(train_records)
    print(f'[train_naive_classifier] fit done in {time.time() - started_at:.2f}s')

    evaluation_metrics = evaluate(trained_model, test_records)
    print(f'[train_naive_classifier] test accuracy: {evaluation_metrics["accuracy"]:.4f}')
    for disease_name, class_stats in evaluation_metrics['per_class'].items():
        print(
            f'  {disease_name}: '
            f'precision={class_stats["precision"]:.3f}  '
            f'recall={class_stats["recall"]:.3f}  '
            f'n={class_stats["support"]}'
        )

    trained_model.metadata.update({
        'test_accuracy':    evaluation_metrics['accuracy'],
        'per_class':        evaluation_metrics['per_class'],
        'confusion_matrix': evaluation_metrics['confusion_matrix'],
        'n_test_samples':   evaluation_metrics['n_test_samples'],
        'training_seconds': round(time.time() - started_at, 3),
    })

    saved_model_path = _model_path(plant_part)
    trained_model.save(saved_model_path)
    invalidate_naive_cache(plant_part)
    print(f'[train_naive_classifier] saved -> {saved_model_path}')

    return {
        'success':    True,
        'plant_part': plant_part,
        'n_train':    len(train_records),
        'n_test':     len(test_records),
        'accuracy':   evaluation_metrics['accuracy'],
        'per_class':  evaluation_metrics['per_class'],
        'model_path': str(saved_model_path),
    }


def get_data_summary_naive(plant_part: str) -> dict:
    """Return readiness summary without training — used by retrain_views."""
    all_records = _fetch_records(plant_part)
    disease_sample_counts = Counter(disease_name for _symptoms, disease_name in all_records)
    eligible_classes = {
        disease_name: sample_count
        for disease_name, sample_count in disease_sample_counts.items()
        if sample_count >= MIN_SAMPLES_PER_CLASS
    }
    return {
        'plant_part':             plant_part,
        'all_classes':            dict(disease_sample_counts),
        'eligible_classes':       eligible_classes,
        'total_eligible_samples': sum(eligible_classes.values()),
        'min_samples_per_class':  MIN_SAMPLES_PER_CLASS,
        'can_train':              len(eligible_classes) >= 2,
    }
