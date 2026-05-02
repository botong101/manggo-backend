"""
Seed description and treatment text into existing Disease rows.

Uses the hardcoded values that were previously in ml_views.py.
Safe to re-run — only updates rows whose treatment field is currently blank.

Run with:
    cd manggo-backend
    python seed_disease_treatments.py
"""

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mangoAPI.settings')
django.setup()

from mangosense.models import Disease

# ---------------------------------------------------------------------------
# Source data (taken verbatim from the treatment_suggestions dict in ml_views)
# Keys match Disease.name values in the DB.
# ---------------------------------------------------------------------------

TREATMENTS = {
    'Anthracnose': (
        'The diseased twigs should be pruned and burnt along with fallen leaves. '
        'Spraying twice with Carbendazim (Bavistin 0.1%) at 15 days interval '
        'during flowering controls blossom infection.',
    ),
    'Bacterial Canker': (
        'Three sprays of Streptocycline (0.01%) or Agrimycin-100 (0.01%) after '
        'first visual symptom at 10 day intervals are effective in controlling '
        'the disease.',
    ),
    'Cutting Weevil': (
        'Use recommended insecticides and remove infested plant material.',
    ),
    'Die Back': (
        'Pruning of the diseased twigs 2-3 inches below the affected portion and '
        'spraying Copper Oxychloride (0.3%) on infected trees controls the disease.',
    ),
    'Gall Midge': (
        'Remove and destroy infested fruits; use appropriate insecticides.',
    ),
    'Healthy': (
        'No treatment needed. Maintain good agricultural practices.',
    ),
    'Powdery Mildew': (
        'Alternate spraying of Wettable sulphur 0.2 per cent at 15 days interval '
        'are recommended for effective control of the disease.',
    ),
    'Sooty Mold': (
        'Pruning of affected branches and their prompt destruction followed by '
        'spraying of Wettasulf (0.2%) helps to control the disease.',
    ),
    'Black Mold Rot': (
        'Improve air circulation around fruit and apply approved fungicides '
        'as needed. Avoid mechanical damage during harvest.',
    ),
    'Stem End Rot': (
        'Proper post-harvest handling and cool storage conditions are essential. '
        'Fungicide dips immediately after harvest can reduce infection rates.',
    ),
    'Alternaria': (
        'Remove and destroy infected fruit. Apply copper-based fungicides '
        'during the growing season. Ensure good ventilation in storage.',
    ),
}

DESCRIPTIONS = {
    'Anthracnose': (
        'A fungal disease caused by Colletotrichum gloeosporioides that affects '
        'mango leaves, flowers, and fruits, producing dark sunken lesions.'
    ),
    'Bacterial Canker': (
        'A bacterial infection causing water-soaked lesions that turn brown '
        'and develop into cankers on leaves, stems, and fruit.'
    ),
    'Cutting Weevil': (
        'Insect pest (Deporaus marginatus) that cuts through leaf stalks, '
        'causing young leaves and shoots to fall prematurely.'
    ),
    'Die Back': (
        'A fungal disease (Lasiodiplodia theobromae) causing progressive '
        'death of twigs and branches from the tip backward.'
    ),
    'Gall Midge': (
        'Infestation by mango gall midge (Erosomyia mangiferae) causing '
        'abnormal gall formations on leaves and stems.'
    ),
    'Healthy': (
        'No disease detected. The mango plant appears healthy with no '
        'visible symptoms of infection or pest damage.'
    ),
    'Powdery Mildew': (
        'Fungal disease (Oidium mangiferae) producing white powdery growth '
        'on young leaves, flowers, and fruits during cool humid weather.'
    ),
    'Sooty Mold': (
        'A black sooty fungal coating that grows on the honeydew secreted '
        'by sucking insects such as aphids, scales, and mealybugs.'
    ),
    'Black Mold Rot': (
        'Post-harvest fruit rot caused by Aspergillus niger, producing '
        'dense black spore masses on the fruit surface.'
    ),
    'Stem End Rot': (
        'Post-harvest rot beginning at the stem end of the fruit, caused '
        'by Lasiodiplodia theobromae or related fungi.'
    ),
    'Alternaria': (
        'Fungal disease caused by Alternaria alternata producing small '
        'dark spots with yellow halos on fruit skin and internal blackening.'
    ),
}

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

updated = 0
skipped = 0
not_found = []

for name, (treatment,) in TREATMENTS.items():
    description = DESCRIPTIONS.get(name, '')

    # match case-insensitively across both plant parts
    qs = Disease.objects.filter(name__iexact=name)
    if not qs.exists():
        not_found.append(name)
        continue

    for disease in qs:
        changed = False
        if not disease.treatment:
            disease.treatment = treatment
            changed = True
        if not disease.description and description:
            disease.description = description
            changed = True
        if changed:
            disease.save(update_fields=['treatment', 'description'])
            updated += 1
            print(f'  [+] Updated: {disease.plant_part}:{disease.name}')
        else:
            skipped += 1
            print(f'  [=] Already set: {disease.plant_part}:{disease.name}')

print(f'\nDone. Updated: {updated}, Already set: {skipped}')
if not_found:
    print(f'No Disease row found for: {not_found}')
    print('  → Create those rows in the admin or run the model seed first.')
