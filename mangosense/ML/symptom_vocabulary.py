
# ── Leaf symptoms (22) ────────────────────────────────────────────────────────
LEAF_SYMPTOMS = [
    # Anthracnose
    'dark_spots_brown',
    'dark_spots_with_yellow_halo',
    'concentric_rings_on_lesion',
    'black_specks_in_lesion',
    'pink_spore_masses',
    'lesions_enlarge_rapidly',
    'spots_with_irregular_border',
    # Die Back
    'twig_dieback_from_tip',
    'canker_lesions_on_twig',
    'bark_cracking',
    'wilting_shoot_tips',
    'sparse_foliage',
    # Powdery Mildew
    'white_powder_coating',
    'leaf_distortion',
    'premature_leaf_drop',
    # Sooty Mold
    'black_sooty_coating',
    'sooty_deposit_wiped_off',
    # General / cross-disease
    'yellow_discoloration',
    'brown_leaf_margins',
    'brown_leaf_tips',
    'water_soaked_lesions',
    'leaf_curling',
]

# ── Fruit symptoms (12) ───────────────────────────────────────────────────────
FRUIT_SYMPTOMS = [
    # Anthracnose
    'black_sunken_lesions',
    'brown_patches_spreading',
    'surface_cracks_radiating',
    'pink_spore_masses_on_lesion',
    'lesion_ring_pattern',
    # Rot / general
    'soft_rot_spreading',
    'stem_end_rot',
    'shriveling_of_fruit',
    'premature_fruit_drop',
    # Surface coatings
    'white_powder_on_fruit',
    'black_sooty_deposit_on_fruit',
    'fruit_discoloration',
]

# ── Disease class names (must match LEAF_CLASS_NAMES / FRUIT_CLASS_NAMES in ml_views.py)
LEAF_DISEASES  = ['Anthracnose', 'Die Back', 'Healthy', 'Powdery Mildew', 'Sooty Mold']
FRUIT_DISEASES = ['Anthracnose', 'Healthy']

# ── Alias table (map legacy / variant strings to canonical names) ─────────────
# Expanded from bulk scraping of Purdue Horticulture Morton Mango reference,
# UC IPM Mango factsheets, CABI Compendium, and mobile app getDiseaseSymptoms v1.
# Sources confirm: "twig dieback", "gummosis", "chlorosis", "acervuli",
# "honeydew excreted by pests", "die-back of terminal shoots" as standard terms.
SYMPTOM_ALIASES: dict[str, str] = {

    # ── Anthracnose — leaf ────────────────────────────────────────────────────
    'dark_spots':                       'dark_spots_brown',
    'black_spots':                      'dark_spots_brown',
    'brown_spots':                      'dark_spots_brown',
    'irregular_brown_spots':            'dark_spots_brown',   # mobile app text
    'leaf_spots':                       'dark_spots_brown',
    'necrotic_spots':                   'dark_spots_brown',
    'yellow_halo':                      'dark_spots_with_yellow_halo',
    'halo_spots':                       'dark_spots_with_yellow_halo',
    'yellow_halo_spots':                'dark_spots_with_yellow_halo',
    'concentric_rings':                 'concentric_rings_on_lesion',
    'ring_lesion':                      'concentric_rings_on_lesion',
    'black_specks':                     'black_specks_in_lesion',
    'acervuli':                         'black_specks_in_lesion',   # technical: fruiting bodies
    'dark_specks':                      'black_specks_in_lesion',
    'pink_spores':                      'pink_spore_masses',
    'orange_spore_masses':              'pink_spore_masses',         # mobile: "pink or orange"
    'salmon_spore_masses':              'pink_spore_masses',
    'spore_masses':                     'pink_spore_masses',
    'expanding_lesions':                'lesions_enlarge_rapidly',
    'coalescing_lesions':               'lesions_enlarge_rapidly',
    'spreading_lesions':                'lesions_enlarge_rapidly',
    'lesions_with_dark_borders':        'spots_with_irregular_border',  # mobile app text
    'dark_border_lesions':              'spots_with_irregular_border',
    'irregular_border':                 'spots_with_irregular_border',

    # ── Die Back — leaf ───────────────────────────────────────────────────────
    'twig_dieback':                     'twig_dieback_from_tip',   # Purdue ref text
    'dieback':                          'twig_dieback_from_tip',
    'die_back':                         'twig_dieback_from_tip',
    'shoot_dieback':                    'twig_dieback_from_tip',
    'tip_dieback':                      'twig_dieback_from_tip',
    'branch_dieback':                   'twig_dieback_from_tip',
    'twig_blight':                      'twig_dieback_from_tip',
    'progressive_dieback':              'twig_dieback_from_tip',    # mobile: "Progressive death"
    'progressive_death_of_branches':    'twig_dieback_from_tip',
    'gummosis':                         'bark_cracking',            # Purdue: "gummosis of twigs"
    'resinous_exudate':                 'bark_cracking',
    'bark_splitting':                   'bark_cracking',            # mobile: "cracking or splitting"
    'bark_crack':                       'bark_cracking',
    'wilting_tips':                     'wilting_shoot_tips',
    'die_back_terminal_shoots':         'wilting_shoot_tips',       # Purdue: "die-back of terminal shoots"
    'terminal_shoot_death':             'wilting_shoot_tips',
    'sparse_canopy':                    'sparse_foliage',
    'thin_canopy':                      'sparse_foliage',

    # ── Powdery Mildew — leaf ────────────────────────────────────────────────
    'white_powder':                     'white_powder_coating',
    'white_powdery_coating':            'white_powder_coating',    # mobile: "White, powdery coating"
    'powdery_white_coating':            'white_powder_coating',
    'white_mealy_coating':              'white_powder_coating',    # scraped: "mealy deposit"
    'powdery_growth':                   'white_powder_coating',
    'oidium':                           'white_powder_coating',    # pathogen genus name
    'white_mycelium':                   'white_powder_coating',    # technical
    'leaf_distortion':                  'leaf_distortion',
    'leaf_puckering':                   'leaf_distortion',
    'distorted_growth':                 'leaf_distortion',
    'leaf_curling_and_distortion':      'leaf_distortion',         # mobile app text
    'leaf_curl':                        'leaf_curling',

    # ── Sooty Mould — leaf ───────────────────────────────────────────────────
    'black_coating':                    'black_sooty_coating',
    'sooty_coating':                    'black_sooty_coating',
    'black_sooty_mold':                 'black_sooty_coating',
    'black_sooty_mould':                'black_sooty_coating',
    'sooty_mold':                       'black_sooty_coating',     # US spelling
    'sooty_mould':                      'black_sooty_coating',     # UK spelling
    'honeydew_coating':                 'black_sooty_coating',     # Purdue: "grows on honeydew"
    'fungal_coating':                   'black_sooty_coating',
    'honeydew':                         'sooty_deposit_wiped_off', # Purdue: "honeydew excreted by pests"
    'wiped_off':                        'sooty_deposit_wiped_off',
    'rubs_off':                         'sooty_deposit_wiped_off',
    'sticky_coating':                   'sooty_deposit_wiped_off',

    # ── General leaf symptoms ─────────────────────────────────────────────────
    'yellowing':                        'yellow_discoloration',
    'chlorosis':                        'yellow_discoloration',    # Purdue: "chlorosis in young trees"
    'chlorotic':                        'yellow_discoloration',
    'yellow_leaves':                    'yellow_discoloration',
    'brown_margins':                    'brown_leaf_margins',
    'yellowing_of_leaf_margins':        'brown_leaf_margins',      # mobile: "Yellowing of leaf margins"
    'marginal_chlorosis':               'brown_leaf_margins',
    'brown_tips':                       'brown_leaf_tips',
    'browning_of_leaf_tips':            'brown_leaf_tips',         # Purdue: "browning of the leaf tips"
    'tip_burn':                         'brown_leaf_tips',
    'water_soaked':                     'water_soaked_lesions',
    'water_soaked_spots':               'water_soaked_lesions',    # mobile: "Water-soaked spots"
    'greasy_spots':                     'water_soaked_lesions',
    'angular_lesions':                  'water_soaked_lesions',    # Bacterial Canker characteristic
    'leaf_drop':                        'premature_leaf_drop',
    'premature_drop':                   'premature_leaf_drop',
    'defoliation':                      'premature_leaf_drop',
    'defoliation_from_tips':            'premature_leaf_drop',     # mobile app text

    # ── Fruit symptoms ───────────────────────────────────────────────────────
    'dark_sunken_spots':                'black_sunken_lesions',    # mobile: "Dark, sunken spots"
    'sunken_lesions':                   'black_sunken_lesions',
    'circular_lesions':                 'black_sunken_lesions',
    'black_sunken':                     'black_sunken_lesions',
    'brown_patches':                    'brown_patches_spreading',
    'spreading_rot':                    'soft_rot_spreading',      # mobile: "Soft rot spreading"
    'internal_rot':                     'soft_rot_spreading',
    'soft_rot':                         'soft_rot_spreading',
    'stem_rot':                         'stem_end_rot',
    'stem_end_infection':               'stem_end_rot',
    'crown_rot':                        'stem_end_rot',
    'shriveled_fruit':                  'shriveling_of_fruit',
    'mummified_fruit':                  'shriveling_of_fruit',
    'fruit_drop':                       'premature_fruit_drop',
    'surface_cracks':                   'surface_cracks_radiating',
    'radiating_cracks':                 'surface_cracks_radiating',
    'ring_pattern':                     'lesion_ring_pattern',
    'concentric_circles_on_fruit':      'lesion_ring_pattern',
    'sooty_fruit':                      'black_sooty_deposit_on_fruit',
    'white_powder_fruit':               'white_powder_on_fruit',
    'white_powdery_fruit':              'white_powder_on_fruit',   # mobile: "White powder on fruit"
    'discolored_fruit':                 'fruit_discoloration',
    'fruit_staining':                   'fruit_discoloration',
    'skin_blemishes':                   'fruit_discoloration',
}


def normalize_symptom(s: str) -> str:
    """Lowercase, strip, and resolve alias → canonical name."""
    cleaned = s.lower().strip().replace(' ', '_').replace('-', '_')
    return SYMPTOM_ALIASES.get(cleaned, cleaned)


def get_vocabulary(disease_type: str) -> list[str]:
    """Return the symptom list for 'leaf' or 'fruit'."""
    return LEAF_SYMPTOMS if disease_type == 'leaf' else FRUIT_SYMPTOMS


def get_diseases(disease_type: str) -> list[str]:
    """Return the disease class list for 'leaf' or 'fruit'."""
    return LEAF_DISEASES if disease_type == 'leaf' else FRUIT_DISEASES