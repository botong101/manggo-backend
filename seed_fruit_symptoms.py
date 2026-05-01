"""
Seed symptoms for Alternaria, Stem End Rot, and Black Mold Rot (fruit diseases).

Run with:
    cd manggo-backend
    python manage.py shell < seed_fruit_symptoms.py
"""

import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mangoAPI.settings')
django.setup()

from mangosense.models import Symptom, Disease, DiseaseSymptom

# ---------------------------------------------------------------------------
# 1. Symptom definitions  (key, display label per disease defined below)
# ---------------------------------------------------------------------------

FRUIT_SYMPTOMS = [
    'small_dark_spots',
    'yellow_halo_around_lesion',
    'internal_black_discoloration',
    'sunken_lesions_fruit',
    'lesions_coalesce',
    'dark_spores_on_lesion',
    'premature_fruit_drop',
    'stem_end_browning',
    'rot_spreads_downward',
    'water_soaked_near_stalk',
    'internal_pulp_browning',
    'white_mycelium_stem',
    'fruit_shriveling_stem',
    'skin_darkening',
    'black_powdery_mold',
    'black_spore_masses',
    'soft_mushy_texture',
    'starts_at_wounds',
    'white_mycelium_beneath',
    'rapid_lesion_expansion',
    'musty_odor',
]

# ---------------------------------------------------------------------------
# 2. Disease → symptom links  {disease_name: [(key, display_label, order), ...]}
# ---------------------------------------------------------------------------

DISEASE_SYMPTOMS = {
    'Alternaria': [
        ('small_dark_spots',           'Small dark brown to black spots on fruit skin',             0),
        ('yellow_halo_around_lesion',  'Yellow or light-colored halo around lesions',               1),
        ('internal_black_discoloration','Internal black discoloration inside fruit',                 2),
        ('sunken_lesions_fruit',       'Sunken lesions on fruit surface',                           3),
        ('lesions_coalesce',           'Spots enlarge and merge into larger patches',               4),
        ('dark_spores_on_lesion',      'Dark powdery spores on lesion surface in humid conditions', 5),
        ('premature_fruit_drop',       'Premature fruit drop when severely affected',               6),
    ],
    'Stem end Rot': [
        ('stem_end_browning',          'Brown to black discoloration starting at the stem end',     0),
        ('rot_spreads_downward',       'Rot spreads progressively from stem end toward the middle', 1),
        ('water_soaked_near_stalk',    'Soft, water-soaked lesions near the stalk',                 2),
        ('internal_pulp_browning',     'Internal browning and softening of pulp',                   3),
        ('white_mycelium_stem',        'White mycelium visible near stem in humid conditions',      4),
        ('fruit_shriveling_stem',      'Fruit shriveling and collapsing near the stem end',         5),
        ('skin_darkening',             'Dark skin discoloration progressing toward the equator',    6),
    ],
    'Black Mold Rot': [
        ('black_powdery_mold',         'Black powdery mold growth on fruit surface',                0),
        ('black_spore_masses',         'Dense black spore masses visible on surface',               1),
        ('soft_mushy_texture',         'Soft and mushy texture beneath the mold',                   2),
        ('starts_at_wounds',           'Infection starts at bruises, wounds, or cracks',            3),
        ('white_mycelium_beneath',     'White to cream mycelium visible beneath black spores',      4),
        ('rapid_lesion_expansion',     'Spots rapidly expand and cover large areas',                5),
        ('musty_odor',                 'Musty or moldy odor from affected fruit',                   6),
    ],
}

# ---------------------------------------------------------------------------
# 3. Create symptoms
# ---------------------------------------------------------------------------

created_symptoms = 0
skipped_symptoms = 0

for key in FRUIT_SYMPTOMS:
    obj, created = Symptom.objects.get_or_create(
        key=key,
        plant_part='fruit',
        defaults={'is_in_vocabulary': True}
    )
    if created:
        created_symptoms += 1
        print(f"  [+] Symptom created: {key}")
    else:
        skipped_symptoms += 1
        print(f"  [=] Symptom exists:  {key}")

print(f"\nSymptoms: {created_symptoms} created, {skipped_symptoms} already existed.\n")

# ---------------------------------------------------------------------------
# 4. Create disease-symptom links
# ---------------------------------------------------------------------------

created_links = 0
skipped_links = 0
missing_diseases = []

for disease_name, symptom_list in DISEASE_SYMPTOMS.items():
    try:
        disease = Disease.objects.get(name=disease_name, plant_part='fruit')
    except Disease.DoesNotExist:
        # Create the disease if it doesn't exist yet
        disease = Disease.objects.create(
            name=disease_name,
            plant_part='fruit',
            is_in_classifier=True
        )
        print(f"  [+] Disease created: {disease_name}")

    for key, label, order in symptom_list:
        try:
            symptom = Symptom.objects.get(key=key, plant_part='fruit')
        except Symptom.DoesNotExist:
            print(f"  [!] Symptom not found: {key} — skipping link")
            continue

        obj, created = DiseaseSymptom.objects.get_or_create(
            disease=disease,
            symptom=symptom,
            defaults={'display_label': label, 'display_order': order}
        )
        if created:
            created_links += 1
            print(f"  [+] Link: {disease_name} ← {key}")
        else:
            skipped_links += 1
            print(f"  [=] Link exists: {disease_name} ← {key}")

print(f"\nLinks: {created_links} created, {skipped_links} already existed.")
print("\nDone.")
