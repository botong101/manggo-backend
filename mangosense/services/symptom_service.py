
from __future__ import annotations

from mangosense.models import Disease, DiseaseSymptom, Symptom, SymptomAlias
from mangosense.repositories.symptom_repository import (
    create_alias,
    create_disease,
    create_disease_symptom,
    create_symptom,
    delete_alias,
    delete_disease,
    delete_disease_symptom,
    delete_symptom,
    display_order_conflict_exists,
    invalidate_symptom_cache,
    symptom_has_disease_links,
    update_alias,
    update_disease,
    update_disease_symptom,
    update_symptom,
)


# symptom

def service_create_symptom(validated_data: dict) -> Symptom:
    symptom = create_symptom(validated_data)
    invalidate_symptom_cache()
    return symptom


def service_update_symptom(symptom: Symptom, validated_data: dict) -> Symptom:
    symptom = update_symptom(symptom, validated_data)
    invalidate_symptom_cache()
    return symptom


def service_delete_symptom(symptom: Symptom) -> None:
    """Raises ValueError if the symptom is still linked to any disease."""
    if symptom_has_disease_links(symptom):
        raise ValueError(
            "Cannot delete: this symptom is linked to one or more diseases. "
            "Remove those DiseaseSymptom links first."
        )
    delete_symptom(symptom)
    invalidate_symptom_cache()


# symptomAlias

def service_create_alias(validated_data: dict) -> SymptomAlias:
    alias = create_alias(validated_data)
    invalidate_symptom_cache()
    return alias


def service_update_alias(alias: SymptomAlias, validated_data: dict) -> SymptomAlias:
    alias = update_alias(alias, validated_data)
    invalidate_symptom_cache()
    return alias


def service_delete_alias(alias: SymptomAlias) -> None:
    delete_alias(alias)
    invalidate_symptom_cache()


# disease

def service_create_disease(validated_data: dict) -> Disease:
    disease = create_disease(validated_data)
    invalidate_symptom_cache()
    return disease


def service_update_disease(disease: Disease, validated_data: dict) -> Disease:
    disease = update_disease(disease, validated_data)
    invalidate_symptom_cache()
    return disease


def service_delete_disease(disease: Disease) -> None:
    # Disease FK uses CASCADE — DiseaseSymptom rows are removed automatically
    # by the DB. No guard needed here; the cascade is intentional.
    delete_disease(disease)
    invalidate_symptom_cache()


# diseaseSymptom 

def service_create_disease_symptom(validated_data: dict) -> DiseaseSymptom:
    """Raises ValueError if display_order already exists for the same disease."""
    disease_id = validated_data.get('disease_id') or validated_data['disease'].pk
    display_order = validated_data['display_order']

    if display_order_conflict_exists(disease_id, display_order):
        raise ValueError(
            f"display_order {display_order} is already taken for this disease. "
            "Choose a different order value or reorder existing rows first."
        )

    link = create_disease_symptom(validated_data)
    invalidate_symptom_cache()
    return link


def service_update_disease_symptom(link: DiseaseSymptom, validated_data: dict) -> DiseaseSymptom:
    """Raises ValueError on display_order collision with a different row."""
    if 'display_order' in validated_data:
        disease_id = validated_data.get('disease_id', link.disease_id)
        new_order = validated_data['display_order']

        if display_order_conflict_exists(disease_id, new_order, exclude_pk=link.pk):
            raise ValueError(
                f"display_order {new_order} is already taken for this disease. "
                "Use a temporary value (e.g. 999) when swapping two rows."
            )

    link = update_disease_symptom(link, validated_data)
    invalidate_symptom_cache()
    return link


def service_delete_disease_symptom(link: DiseaseSymptom) -> None:
    delete_disease_symptom(link)
    invalidate_symptom_cache()