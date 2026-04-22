from rest_framework.decorators import api_view
from django.http import JsonResponse

# Each symptom entry is {"key": <canonical vocabulary key>, "label": <display string>}.
# Keys for diseases in LEAF_DISEASES / FRUIT_DISEASES (Anthracnose, Die Back,
# Powdery Mildew, Sooty Mould, Healthy) map directly to LEAF_SYMPTOMS / FRUIT_SYMPTOMS
# in symptom_vocabulary.py so the XGBoost encoder produces non-zero feature vectors.
# Keys for other diseases (Bacterial Canker, Cutting Weevil, Gall Midge) are descriptive
# snake_case strings stored in the DB for future use when those diseases are added to
# the classifier vocabulary.

def _s(key: str, label: str) -> dict:
    return {'key': key, 'label': label}


SYMPTOMS_MAP = {
    # ── Anthracnose ───────────────────────────────────────────────────────────
    'Anthracnose': {
        'fruit': [
            _s('black_sunken_lesions',       'Dark, sunken spots on fruit surface'),
            _s('pink_spore_masses_on_lesion','Pink or salmon spore masses in humid conditions'),
            _s('premature_fruit_drop',       'Premature fruit drop before ripening'),
            _s('brown_patches_spreading',    'Lesions enlarge and coalesce during ripening'),
            _s('soft_rot_spreading',         'Soft rot develops beneath surface lesions'),
            _s('surface_cracks_radiating',   'Skin cracks radiating from expanding lesions'),
        ],
        'leaf': [
            _s('dark_spots_brown',              'Irregular brown or black spots on leaves'),
            _s('dark_spots_with_yellow_halo',   'Lesions with dark borders and yellow halo'),
            _s('pink_spore_masses',             'Pink or orange spore masses on moist lesions'),
            _s('yellow_discoloration',          'Yellowing around affected areas'),
            _s('premature_leaf_drop',           'Premature leaf drop'),
            _s('black_specks_in_lesion',        'Black specks (acervuli) within older lesions'),
            _s('lesions_enlarge_rapidly',       'Spots enlarge and coalesce rapidly under wet conditions'),
        ],
    },

    # ── Bacterial Canker ──────────────────────────────────────────────────────
    # Not in current XGBoost vocabulary; keys stored for future classifier expansion.
    'Bacterial Canker': {
        'fruit': [
            _s('canker_branch_dieback',    'Ends of branches drying up and dying back'),
            _s('canker_tip_progression',   'Drying progressing from tip toward main branch'),
            _s('canker_plant_stress',      'Plant appears generally stressed and weak'),
            _s('canker_gummosis',          'Gummosis or resinous exudate on stems'),
            _s('canker_reduced_fruit_set', 'Reduced fruit set on affected branches'),
        ],
        'leaf': [
            _s('water_soaked_lesions',  'Water-soaked, greasy-looking spots on leaves'),
            _s('water_soaked_lesions',  'Angular lesions limited by leaf veins'),
            _s('brown_leaf_margins',    'Yellowing of leaf margins'),
            _s('brown_leaf_tips',       'Brown, necrotic dead areas on leaf blade'),
            _s('premature_leaf_drop',   'Leaf wilting followed by drop'),
            _s('water_soaked_lesions',  'Dark water-soaked streaks along midrib'),
        ],
    },

    # ── Cutting Weevil ────────────────────────────────────────────────────────
    # Not in current XGBoost vocabulary; keys stored for future classifier expansion.
    'Cutting Weevil': {
        'fruit': [
            _s('weevil_entry_holes',    'Small entry holes in young shoots and tender leaves'),
            _s('weevil_shoot_wilt',     'Wilting of terminal shoots shortly after attack'),
            _s('weevil_insect_present', 'Presence of small reddish-brown weevil insects'),
            _s('weevil_growing_tips',   'Damage concentrated at actively growing tips'),
            _s('weevil_frass',          'Frass or boring dust around entry holes'),
        ],
        'leaf': [
            _s('brown_leaf_tips',     'Browning and drying of leaves from tip backward'),
            _s('yellow_discoloration','Yellowing preceding browning of leaf tissue'),
            _s('leaf_curling',        'Leaf curling and wilting of affected leaves'),
            _s('premature_leaf_drop', 'Defoliation starting from branch tips downward'),
            _s('weevil_cut_petioles', 'Cut or severed petioles at the base of leaves'),
            _s('weevil_ragged_flush', 'Young flush leaves appear ragged or notched'),
        ],
    },

    # ── Die Back ──────────────────────────────────────────────────────────────
    'Die Back': {
        'fruit': [
            _s('twig_dieback_from_tip',  'Progressive death of branches from tips downward'),
            _s('brown_leaf_tips',        'Browning and drying of attached leaves on affected shoots'),
            _s('bark_cracking',          'Bark cracking or splitting on affected twigs'),
            _s('canker_lesions_on_twig', 'Dark sunken canker lesions on twigs and branches'),
            _s('bark_cracking',          'Gummosis from cracked bark'),
            _s('fruit_discoloration',    'Reduced fruit production on symptomatic branches'),
        ],
        'leaf': [
            _s('brown_leaf_tips',       'Browning and drying of leaves starting at tip'),
            _s('yellow_discoloration',  'Yellowing before browning of leaf tissue'),
            _s('leaf_curling',          'Leaf curling and wilting'),
            _s('premature_leaf_drop',   'Defoliation starting from branch tips'),
            _s('twig_dieback_from_tip', 'Twig dieback progressing toward main stem'),
            _s('sparse_foliage',        'Sparse foliage on affected branches'),
            _s('wilting_shoot_tips',    'Wilting of terminal shoot tips'),
        ],
    },

    # ── Gall Midge ────────────────────────────────────────────────────────────
    # Not in current XGBoost vocabulary; keys stored for future classifier expansion.
    'Gall Midge': {
        'fruit': [
            _s('gall_bumps_shoots',    'Small bumps or galls on leaf surface near affected shoots'),
            _s('leaf_distortion',      'Distorted and puckered leaf growth on infested flush'),
            _s('gall_maggots',         'Tiny orange maggots inside galls when opened'),
            _s('gall_midge_present',   'Presence of small midges around young flush'),
            _s('wilting_shoot_tips',   'Stunted and distorted shoot development'),
        ],
        'leaf': [
            _s('gall_leaf_bumps',    'Small raised bumps or galls on leaf surface'),
            _s('leaf_distortion',    'Distorted, twisted and crinkled leaf growth'),
            _s('yellow_discoloration','Yellowing around gall areas on leaves'),
            _s('gall_stunted_leaf',  'Stunted leaf expansion and development'),
            _s('gall_leaf_blisters', 'Raised blister-like swellings along leaf margins'),
            _s('premature_leaf_drop','Premature leaf drop of heavily infested flush'),
        ],
    },

    # ── Healthy ───────────────────────────────────────────────────────────────
    # Healthy keys intentionally do NOT appear in LEAF_SYMPTOMS / FRUIT_SYMPTOMS.
    # The XGBoost encoder produces an all-zero vector for healthy plants (no disease
    # symptoms present), which is the correct representation for the Healthy class.
    'Healthy': {
        'fruit': [
            _s('healthy_fruit_development', 'Normal, uniform fruit development and sizing'),
            _s('healthy_fruit_colour',      'Uniform surface colour with no spots or discolouration'),
            _s('healthy_fruit_texture',     'Smooth, firm skin texture with no lesions'),
            _s('healthy_no_blemishes',      'No visible spots, lesions, or surface blemishes'),
            _s('healthy_fruit_retention',   'Normal fruit set and retention on tree'),
        ],
        'leaf': [
            _s('healthy_leaf_colour',   'Vibrant, uniform dark-green leaf colour'),
            _s('healthy_leaf_surface',  'Smooth, unblemished leaf surface'),
            _s('healthy_leaf_size',     'Normal leaf size and shape for variety'),
            _s('healthy_no_lesions',    'No discolouration, spots, or lesions present'),
            _s('healthy_flush_colour',  'New flush emerging with normal pinkish-red colouration'),
            _s('healthy_leaf_firm',     'Leaves held firmly without wilting or curling'),
        ],
    },

    # ── Powdery Mildew ────────────────────────────────────────────────────────
    'Powdery Mildew': {
        'fruit': [
            _s('white_powder_on_fruit',  'White, powdery fungal coating on young fruit surface'),
            _s('fruit_discoloration',    'Yellowing or russeting of affected fruit skin'),
            _s('shriveling_of_fruit',    'Distorted or stunted fruit development'),
            _s('premature_fruit_drop',   'Premature fruit drop of affected young fruits'),
            _s('white_powder_on_fruit',  'White mealy deposit on flower panicles'),
            _s('fruit_discoloration',    'Russeting or scarring on skin of surviving fruit'),
        ],
        'leaf': [
            _s('white_powder_coating', 'White, powdery fungal coating on leaf surfaces'),
            _s('white_powder_coating', 'White mealy or dusty growth, especially on young leaves'),
            _s('yellow_discoloration', 'Yellowing of affected leaves'),
            _s('leaf_distortion',      'Leaf curling, distortion and puckering'),
            _s('premature_leaf_drop',  'Premature leaf drop'),
            _s('brown_leaf_margins',   'Affected leaves turn brown and necrotic at margins'),
            _s('white_powder_coating', 'New flush heavily affected; older leaves less so'),
        ],
    },

    # ── Sooty Mould ───────────────────────────────────────────────────────────
    'Sooty Mould': {
        'fruit': [
            _s('black_sooty_deposit_on_fruit', 'Black, sooty fungal coating on fruit surface'),
            _s('black_sooty_deposit_on_fruit', 'Sticky, shiny residue (honeydew) beneath sooty layer'),
            _s('sooty_deposit_wiped_off',       'Coating rubs off to reveal undamaged skin below'),
            _s('fruit_discoloration',           'Cosmetic damage only; fruit flesh unaffected'),
            _s('black_sooty_deposit_on_fruit', 'Associated with scale insects or mealybug infestations'),
        ],
        'leaf': [
            _s('black_sooty_coating',    'Black, sooty fungal coating on upper leaf surface'),
            _s('black_sooty_coating',    'Coating grows on honeydew excreted by sap-sucking insects'),
            _s('sooty_deposit_wiped_off','Sooty deposit rubs off, revealing green leaf beneath'),
            _s('black_sooty_coating',    'Reduced light absorption and photosynthesis on coated leaves'),
            _s('yellow_discoloration',   'Yellowing of leaves beneath heavy sooty coating'),
            _s('sooty_deposit_wiped_off','Sticky honeydew substance present on leaf surface'),
            _s('black_sooty_coating',    'Presence of scale insects or mealybugs on leaf undersides'),
        ],
    },
}

FALLBACK_SYMPTOMS = [
    _s('obs_discolouration', 'Look for any unusual discolouration or spots'),
    _s('obs_texture',        'Check for changes in texture or firmness'),
    _s('obs_growth',         'Notice any abnormal growth patterns'),
    _s('obs_environment',    'Consider environmental factors affecting the plant'),
]


@api_view(['GET'])
def get_disease_symptoms(request):
    disease = request.query_params.get('disease', '').strip()
    plant_part = request.query_params.get('plant_part', 'leaf').strip().lower()

    if plant_part not in ('leaf', 'fruit'):
        return JsonResponse(
            {'error': "plant_part must be 'leaf' or 'fruit'"},
            status=400
        )

    disease_data = SYMPTOMS_MAP.get(disease)
    if disease_data:
        symptoms = disease_data.get(plant_part, FALLBACK_SYMPTOMS)
    else:
        symptoms = FALLBACK_SYMPTOMS

    return JsonResponse({
        'disease': disease,
        'plant_part': plant_part,
        'symptoms': symptoms,
    })
