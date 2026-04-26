
from mangosense.repositories.symptom_repository import (
    get_alias_map,
    get_diseases,
    get_vocabulary,
    normalize_symptom
)

def __getattr__(name: str):
    if name == 'LEAF_SYMPTOMS':
        return get_vocabulary('leaf')
    if name == 'FRUIT_SYMPTOMS':
        return get_vocabulary('fruit')
    if name == 'LEAF_DISEASES':
        return get_diseases('leaf')
    if name == 'FRUIT_DISEASES':
        return get_diseases('fruit')
    if name == 'SYMPTOM_ALIASES':
        return get_alias_map()
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


""" # ── Leaf symptoms (22) ────────────────────────────────────────────────────────
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

SYMPTOM_ALIASES: dict[str, str] = {

    # ── Anthracnose — leaf ────────────────────────────────────────────────────
    'dark_spots':                       'dark_spots_brown',
    'black_spots':                      'dark_spots_brown',
    'brown_spots':                      'dark_spots_brown',
    'irregular_brown_spots':            'dark_spots_brown',   
    'leaf_spots':                       'dark_spots_brown',
    'necrotic_spots':                   'dark_spots_brown',
    'yellow_halo':                      'dark_spots_with_yellow_halo',
    'halo_spots':                       'dark_spots_with_yellow_halo',
    'yellow_halo_spots':                'dark_spots_with_yellow_halo',
    'concentric_rings':                 'concentric_rings_on_lesion',
    'ring_lesion':                      'concentric_rings_on_lesion',
    'black_specks':                     'black_specks_in_lesion',
    'acervuli':                         'black_specks_in_lesion',   
    'dark_specks':                      'black_specks_in_lesion',
    'pink_spores':                      'pink_spore_masses',
    'orange_spore_masses':              'pink_spore_masses',        
    'salmon_spore_masses':              'pink_spore_masses',
    'spore_masses':                     'pink_spore_masses',
    'expanding_lesions':                'lesions_enlarge_rapidly',
    'coalescing_lesions':               'lesions_enlarge_rapidly',
    'spreading_lesions':                'lesions_enlarge_rapidly',
    'lesions_with_dark_borders':        'spots_with_irregular_border',  
    'dark_border_lesions':              'spots_with_irregular_border',
    'irregular_border':                 'spots_with_irregular_border',

    # ── Die Back — leaf ───────────────────────────────────────────────────────
    'twig_dieback':                     'twig_dieback_from_tip',  
    'dieback':                          'twig_dieback_from_tip',
    'die_back':                         'twig_dieback_from_tip',
    'shoot_dieback':                    'twig_dieback_from_tip',
    'tip_dieback':                      'twig_dieback_from_tip',
    'branch_dieback':                   'twig_dieback_from_tip',
    'twig_blight':                      'twig_dieback_from_tip',
    'progressive_dieback':              'twig_dieback_from_tip',   
    'progressive_death_of_branches':    'twig_dieback_from_tip',
    'gummosis':                         'bark_cracking',           
    'resinous_exudate':                 'bark_cracking',
    'bark_splitting':                   'bark_cracking',           
    'bark_crack':                       'bark_cracking',
    'wilting_tips':                     'wilting_shoot_tips',
    'die_back_terminal_shoots':         'wilting_shoot_tips',       
    'terminal_shoot_death':             'wilting_shoot_tips',
    'sparse_canopy':                    'sparse_foliage',
    'thin_canopy':                      'sparse_foliage',

    # ── Powdery Mildew — leaf ────────────────────────────────────────────────
    'white_powder':                     'white_powder_coating',
    'white_powdery_coating':            'white_powder_coating',    
    'powdery_white_coating':            'white_powder_coating',
    'white_mealy_coating':              'white_powder_coating',    
    'powdery_growth':                   'white_powder_coating',
    'oidium':                           'white_powder_coating',    
    'white_mycelium':                   'white_powder_coating',  
    'leaf_distortion':                  'leaf_distortion',
    'leaf_puckering':                   'leaf_distortion',
    'distorted_growth':                 'leaf_distortion',
    'leaf_curling_and_distortion':      'leaf_distortion',         
    'leaf_curl':                        'leaf_curling',

    # ── Sooty Mould — leaf ───────────────────────────────────────────────────
    'black_coating':                    'black_sooty_coating',
    'sooty_coating':                    'black_sooty_coating',
    'black_sooty_mold':                 'black_sooty_coating',
    'black_sooty_mould':                'black_sooty_coating',
    'sooty_mold':                       'black_sooty_coating',     
    'sooty_mould':                      'black_sooty_coating',     
    'honeydew_coating':                 'black_sooty_coating',     
    'fungal_coating':                   'black_sooty_coating',
    'honeydew':                         'sooty_deposit_wiped_off', 
    'wiped_off':                        'sooty_deposit_wiped_off',
    'rubs_off':                         'sooty_deposit_wiped_off',
    'sticky_coating':                   'sooty_deposit_wiped_off',

    # ── General leaf symptoms ─────────────────────────────────────────────────
    'yellowing':                        'yellow_discoloration',
    'chlorosis':                        'yellow_discoloration',    
    'chlorotic':                        'yellow_discoloration',
    'yellow_leaves':                    'yellow_discoloration',
    'brown_margins':                    'brown_leaf_margins',
    'yellowing_of_leaf_margins':        'brown_leaf_margins',     
    'marginal_chlorosis':               'brown_leaf_margins',
    'brown_tips':                       'brown_leaf_tips',
    'browning_of_leaf_tips':            'brown_leaf_tips',        
    'tip_burn':                         'brown_leaf_tips',
    'water_soaked':                     'water_soaked_lesions',
    'water_soaked_spots':               'water_soaked_lesions',    
    'greasy_spots':                     'water_soaked_lesions',
    'angular_lesions':                  'water_soaked_lesions',   
    'leaf_drop':                        'premature_leaf_drop',
    'premature_drop':                   'premature_leaf_drop',
    'defoliation':                      'premature_leaf_drop',
    'defoliation_from_tips':            'premature_leaf_drop',    

    # ── Fruit symptoms ───────────────────────────────────────────────────────
    'dark_sunken_spots':                'black_sunken_lesions',  
    'sunken_lesions':                   'black_sunken_lesions',
    'circular_lesions':                 'black_sunken_lesions',
    'black_sunken':                     'black_sunken_lesions',
    'brown_patches':                    'brown_patches_spreading',
    'spreading_rot':                    'soft_rot_spreading',     
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
    'white_powdery_fruit':              'white_powder_on_fruit',   
    'discolored_fruit':                 'fruit_discoloration',
    'fruit_staining':                   'fruit_discoloration',
    'skin_blemishes':                   'fruit_discoloration',
}


def normalize_symptom(s: str) -> str:
    cleaned = s.lower().strip().replace(' ', '_').replace('-', '_')
    return SYMPTOM_ALIASES.get(cleaned, cleaned)


def get_vocabulary(disease_type: str) -> list[str]:
    return LEAF_SYMPTOMS if disease_type == 'leaf' else FRUIT_SYMPTOMS


def get_diseases(disease_type: str) -> list[str]:
    return LEAF_DISEASES if disease_type == 'leaf' else FRUIT_DISEASES """