IMG_SIZE = (224, 224)

GATE_LEAF_CLASS_NAMES = [
    'Black plum',
    'Guava',
    'Jackfruit',
    'Lychee',
    'Mango',
    'Plum',
]

GATE_FRUIT_CLASS_NAMES = [
    'Apple',
    'Banana',
    'Grape',
    'Mango',
    'Strawberry',
]

GATE_VALID_INDEX_LEAF = 4
GATE_VALID_INDEX_FRUIT = 3

GATE_CONFIDENCE_THRESHOLD_LEAF = 60.0
GATE_CONFIDENCE_THRESHOLD_FRUIT = 60.0
GATE_CONFIDENCE_THRESHOLD = 60.0

LEAF_CLASS_NAMES = [
    'Anthracnose', 'Die Back', 'Healthy', 'Powdery Mildew', 'Sooty Mold',
]

FRUIT_CLASS_NAMES = [
    'Alternaria', 'Anthracnose', 'Black Mold Rot', 'Healthy', 'Stem end Rot',
]

DISEASE_CONFIDENCE_THRESHOLD = 20.0

_DEFAULT_LEAF_MODEL = 'mobilenetv2-leaf.keras'
_DEFAULT_FRUIT_MODEL = 'mobilenetv2-fruit.keras'
_DEFAULT_GATE_LEAF_MODEL = 'gate-leaf-model-2.keras'
_DEFAULT_GATE_FRUIT_MODEL = 'mango-fruit-vs-others.keras'

# treatment_suggestions = {
#     'Anthracnose': 'hello testing! The diseased twigs should be pruned and burnt along with fallen leaves. Spraying twice with Carbendazim (Bavistin 0.1%) at 15 days interval during flowering controls blossom infection.',
#     'Bacterial Canker': 'Three sprays of Streptocycline (0.01%) or Agrimycin-100 (0.01%) after first visual symptom at 10 day intervals are effective in controlling the disease.',
#     'Cutting Weevil': 'Use recommended insecticides and remove infested plant material.',
#     'Die Back': 'Pruning of the diseased twigs 2-3 inches below the affected portion and spraying Copper Oxychloride (0.3%) on infected trees controls the disease.',
#     'Gall Midge': 'Remove and destroy infested fruits; use appropriate insecticides.',
#     'Healthy': 'No treatment needed. Maintain good agricultural practices.',
#     'Powdery Mildew': 'Alternate spraying of Wettable sulphur 0.2 per cent at 15 days interval are recommended for effective control of the disease.',
#     'Sooty Mold': 'Pruning of affected branches and their prompt destruction followed by spraying of Wettasulf (0.2%) helps to control the disease.',
#     'Black Mold Rot': 'Improve air circulation and apply fungicides as needed.',
#     'Stem End Rot': 'Proper post-harvest handling and storage conditions are essential.',
# }
