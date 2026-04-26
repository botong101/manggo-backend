from django.db import migrations


def seed(apps, schema_editor):
    """Populate all four vocabulary tables from the hardcoded source constants."""

    Symptom        = apps.get_model('mangosense', 'Symptom')
    SymptomAlias   = apps.get_model('mangosense', 'SymptomAlias')
    Disease        = apps.get_model('mangosense', 'Disease')
    DiseaseSymptom = apps.get_model('mangosense', 'DiseaseSymptom')

    from mangosense.ML.symptom_vocabulary import (
        LEAF_SYMPTOMS,
        FRUIT_SYMPTOMS,
        SYMPTOM_ALIASES,
        LEAF_DISEASES,
        FRUIT_DISEASES,
    )
    from mangosense.views.symptom_views import SYMPTOMS_MAP, FALLBACK_SYMPTOMS

    # ── Step 1: Seed Symptom rows from the XGBoost vocabulary ─────────────────
    # vector_index is derived from enumerate() — the position in the list IS
    # the feature-column index. Changing these values invalidates trained models.

    for symptom_index, symptom_key in enumerate(LEAF_SYMPTOMS):
        Symptom.objects.get_or_create(
            key=symptom_key,
            plant_part='leaf',
            defaults={
                'vector_index': symptom_index,
                'is_in_vocabulary': True,
            },
        )

    for symptom_index, symptom_key in enumerate(FRUIT_SYMPTOMS):
        Symptom.objects.get_or_create(
            key=symptom_key,
            plant_part='fruit',
            defaults={
                'vector_index': symptom_index,
                'is_in_vocabulary': True,
            },
        )

    # ── Step 2: Seed Disease rows and DiseaseSymptom links from SYMPTOMS_MAP ──

    leaf_disease_names  = set(LEAF_DISEASES)
    fruit_disease_names = set(FRUIT_DISEASES)

    for disease_name, plant_parts in SYMPTOMS_MAP.items():
        for plant_part, symptom_list in plant_parts.items():

            if plant_part == 'leaf':
                in_classifier = disease_name in leaf_disease_names
            else:
                in_classifier = disease_name in fruit_disease_names

            disease_obj, _ = Disease.objects.get_or_create(
                name=disease_name,
                plant_part=plant_part,
                defaults={'is_in_classifier': in_classifier},
            )

            for display_order, symptom_entry in enumerate(symptom_list):
                symptom_key   = symptom_entry['key']
                display_label = symptom_entry['label']

                # Vocabulary keys were seeded in Step 1. Display-only keys
                # (canker_*, weevil_*, healthy_*) are created here with no vector position.
                symptom_obj, _ = Symptom.objects.get_or_create(
                    key=symptom_key,
                    plant_part=plant_part,
                    defaults={
                        'vector_index': None,
                        'is_in_vocabulary': False,
                    },
                )

                DiseaseSymptom.objects.update_or_create(
                    disease=disease_obj,
                    display_order=display_order,
                    defaults={
                        'symptom': symptom_obj,
                        'display_label': display_label,
                    },
                )

    # ── Step 3: Seed SymptomAlias rows ────────────────────────────────────────
    # Try leaf first; fall back to fruit. Aliases without a matching canonical
    # Symptom are skipped with a warning — a typo should not abort the migration.

    for alias_key, canonical_key in SYMPTOM_ALIASES.items():
        canonical_symptom = (
            Symptom.objects.filter(key=canonical_key, plant_part='leaf').first()
            or Symptom.objects.filter(key=canonical_key, plant_part='fruit').first()
        )

        if canonical_symptom is None:
            print(
                f"[seed_symptom_vocabulary] WARNING: alias '{alias_key}' → "
                f"'{canonical_key}' skipped — canonical key not found in DB."
            )
            continue

        SymptomAlias.objects.get_or_create(
            alias=alias_key,
            defaults={'canonical': canonical_symptom, 'source': 'symptom_vocabulary.py seed'},
        )


def reverse_seed(apps, schema_editor):
    """Remove all seeded rows in FK-safe deletion order."""
    DiseaseSymptom = apps.get_model('mangosense', 'DiseaseSymptom')
    SymptomAlias   = apps.get_model('mangosense', 'SymptomAlias')
    Disease        = apps.get_model('mangosense', 'Disease')
    Symptom        = apps.get_model('mangosense', 'Symptom')

    DiseaseSymptom.objects.all().delete()
    SymptomAlias.objects.all().delete()
    Disease.objects.all().delete()
    Symptom.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('mangosense', '0023_disease_symptom_symptomalias_diseasesymptom'),
    ]

    operations = [
        migrations.RunPython(
            seed,
            reverse_code=reverse_seed,
            elidable=False,
        ),
    ]
