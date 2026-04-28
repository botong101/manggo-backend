from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mangosense.models import Disease, DiseaseSymptom, Symptom, SymptomAlias

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()



def _normalise_plant_part(plant_part: str) -> str:
    """Lowercase and strip the plant_part argument. Raises ValueError on bad input."""
    normalized_part = plant_part.strip().lower()
    if normalized_part not in ('leaf', 'fruit'):
        raise ValueError(f"plant_part must be 'leaf' or 'fruit', got {plant_part!r}")
    return normalized_part


def get_vocabulary(plant_part: str) -> list[str]:
    """Return the ordered list of canonical symptom keys used as XGBoost feature columns.

    Only Symptom rows with is_in_vocabulary=True are included, ordered by
    vector_index ascending. This ordering MUST be stable across calls because
    position == feature column index. The DB is the source of truth for that order.

    Args:
        plant_part: 'leaf' or 'fruit'

    Returns:
        list of canonical key strings, e.g. ['dark_spots_brown', ...]
    """
    from mangosense.models import Symptom  # noqa: PLC0415

    normalized_part = _normalise_plant_part(plant_part)
    cache_key = f'vocabulary:{normalized_part}'

    with _LOCK:
        if cache_key not in _CACHE:
            queryset = (
                Symptom.objects
                .filter(plant_part=normalized_part, is_in_vocabulary=True)
                .exclude(vector_index=None)
                .order_by('vector_index')
                .values_list('key', flat=True)
            )
            _CACHE[cache_key] = list(queryset)
        return _CACHE[cache_key]
    
def get_diseases(plant_part: str) -> list[str]:
    """Return the list of disease names the classifier can predict.

    Only Disease rows with is_in_classifier=True are included, sorted
    alphabetically. Alphabetical order matches the assumption baked into
    the XGBoost training script (LabelEncoder fits sorted class names).

    Args:
        plant_part: 'leaf' or 'fruit'

    Returns:
        list of disease name strings, e.g. ['Anthracnose', 'Die Back', ...]
    """

    from mangosense.models import Disease

    normalized_part = _normalise_plant_part(plant_part)
    cache_key = f'disease:{normalized_part}'

    with _LOCK:
        if cache_key not in _CACHE:
            queryset = (
                Disease.objects
                .filter(plant_part=normalized_part, is_in_classifier=True)
                .order_by('name')
                .values_list('name', flat=True)
            )
            _CACHE[cache_key] = list(queryset)
        return _CACHE[cache_key]
    
def get_alias_map() -> dict[str, str]:
    """Return the full alias -> canonical key mapping for bulk normalisation.

    Used by training notebooks that normalise an entire dataset at once.
    The returned dict is a plain Python dict callers may iterate it freely
    but should NOT mutate it (mutations are not reflected back to the DB).

    Returns:
        dict mapping alias strings to canonical Symptom.key strings.
    """
    from mangosense.models import SymptomAlias

    cache_key = 'aliases'

    with _LOCK:
        if cache_key not in _CACHE:
            queryset = (
                SymptomAlias.objects
                .select_related('canonical')
                .values_list('alias', 'canonical__key')
            )
            _CACHE[cache_key] = {alias_key: canonical_key for alias_key,
                                 canonical_key in queryset}
        return _CACHE[cache_key]
    
def normalize_symptom(raw_symptom: str)->str:
    """Normalise a raw symptom string to its canonical key.

    Steps:
    1. Lowercase + strip whitespace.
    2. Replace spaces and hyphens with underscores.
    3. Look up the result in the alias map. If found, return the canonical key.
    4. If not found, return the normalised string as-is (may be a canonical key
       already, or an unknown symptom — the encoder handles unknowns as all-zero).

    Args:
        raw_symptom: raw symptom string from the mobile app or training data.

    Returns:
        canonical key string.
    """
    normalised_symptom = raw_symptom.strip().lower().replace(' ', '_').replace('-', '_')
    alias_map = get_alias_map()
    return alias_map.get(normalised_symptom, normalised_symptom)

def get_symptoms_for_disease(disease_name: str, plant_part: str) -> list[dict]:
    """Return the [{key, label}] list for a specific disease + plant_part combination.

    Rows come from the DiseaseSymptom join table, ordered by display_order.
    This is the data source for the /api/symptoms/ endpoint.

    Args:
        disease_name: Disease.name, e.g. 'Anthracnose'
        plant_part:   'leaf' or 'fruit'

    Returns:
        list of dicts with keys 'key' and 'label', or empty list if no rows found.
    """

    from mangosense.models import DiseaseSymptom

    normalize_part = _normalise_plant_part(plant_part)

    cache_key = f'symptoms:{disease_name}:{normalize_part}'

    with _LOCK:
        if cache_key not in _CACHE:
            queryset = (
                DiseaseSymptom.objects
                .select_related('symptom', 'disease')
                .filter(disease__name = disease_name, 
                        disease__plant_part = normalize_part)
                .order_by('display_order')
                .values('symptom__key', 'display_label')
            )
            _CACHE[cache_key] = [
                {'key': row['symptom__key'], 'label': row['display_label']}
                for row in queryset
            ]
        return _CACHE[cache_key]
    
def get_fallback_symptoms() ->list[dict]:
    """Return the generic observation prompts shown when no disease is recognised.

    DESIGN DECISION — synthetic Disease='_fallback'
    -----------------------------------------------
    Rather than keeping a hardcoded list-of-dicts in this module (the last
    remaining Python constant), these rows are stored in the DB as DiseaseSymptom
    entries under a Disease with name='_fallback'. The leading underscore is a
    convention that signals "infrastructure row, not a real disease" — the admin
    panel should display it in a separate section. This gives the product team
    full control over the fallback prompts via the Django admin without a code
    deploy.

    Returns:
        list of dicts with keys 'key' and 'label'.
    """
    cache_key = 'fallback'

    with _LOCK:
        if cache_key not in _CACHE:

            from mangosense.models import DiseaseSymptom

            queryset = (
                DiseaseSymptom.objects
                .select_related('symptom', 'disease')
                .filter(disease__name='_fallback')
                .order_by('display_order')
                .values('symptom__key', 'display_label')
            )
            fallback_rows = [
                {'key': row['symptom__key'], 'label': row['display_label']}
                for row in queryset
            ]

            
            if not fallback_rows:
                fallback_rows = [
                    {'key': 'obs_discolouration', 'label': 'Look for any unusual discolouration or spots'},
                    {'key': 'obs_texture',         'label': 'Check for changes in texture or firmness'},
                    {'key': 'obs_growth',           'label': 'Notice any abnormal growth patterns'},
                    {'key': 'obs_environment',      'label': 'Consider environmental factors affecting the plant'},
                ]
            _CACHE[cache_key] = fallback_rows 
            
        return _CACHE[cache_key]

def invalidate_symptom_cache()-> None:
    with _LOCK:
        _CACHE.clear()
    

# symptom write operations

def create_symptom(validated_data: dict) -> Symptom:
    from mangosense.models import Symptom  # noqa: PLC0415
    """Insert one Symptom row. Returns the saved instance."""
    return Symptom.objects.create(**validated_data)


def update_symptom(symptom: Symptom, validated_data: dict) -> Symptom:
    """Apply validated_data fields to symptom and save. Returns updated instance."""
    for field, value in validated_data.items():
        setattr(symptom, field, value)
    symptom.save()
    return symptom


def delete_symptom(symptom: Symptom) -> None:
    """Delete one Symptom row. Caller must have already confirmed no PROTECT links."""
    symptom.delete()


def symptom_has_disease_links(symptom: Symptom) -> bool:
    """Return True if any DiseaseSymptom row references this symptom."""
    return symptom.disease_symptoms.exists()


# symptomAlias write operations

def create_alias(validated_data: dict) -> SymptomAlias:
    from mangosense.models import SymptomAlias  # noqa: PLC0415
    return SymptomAlias.objects.create(**validated_data)


def update_alias(alias: SymptomAlias, validated_data: dict) -> SymptomAlias:
    for field, value in validated_data.items():
        setattr(alias, field, value)
    alias.save()
    return alias


def delete_alias(alias: SymptomAlias) -> None:
    alias.delete()


# disease write operations 

def create_disease(validated_data: dict) -> Disease:
    from mangosense.models import Disease  # noqa: PLC0415
    return Disease.objects.create(**validated_data)


def update_disease(disease: Disease, validated_data: dict) -> Disease:
    for field, value in validated_data.items():
        setattr(disease, field, value)
    disease.save()
    return disease


def delete_disease(disease: Disease) -> None:
    disease.delete()


# diseaseSymptom write operations 

def display_order_conflict_exists(disease_id: int, display_order: int, exclude_pk: int | None = None) -> bool:
    """Return True if another DiseaseSymptom for the same disease already uses this display_order."""
    from mangosense.models import DiseaseSymptom  # noqa: PLC0415
    qs = DiseaseSymptom.objects.filter(disease_id=disease_id, display_order=display_order)
    if exclude_pk is not None:
        qs = qs.exclude(pk=exclude_pk)
    return qs.exists()


def create_disease_symptom(validated_data: dict) -> DiseaseSymptom:
    from mangosense.models import DiseaseSymptom  # noqa: PLC0415
    return DiseaseSymptom.objects.create(**validated_data)


def update_disease_symptom(link: DiseaseSymptom, validated_data: dict) -> DiseaseSymptom:
    for field, value in validated_data.items():
        setattr(link, field, value)
    link.save()
    return link


def delete_disease_symptom(link: DiseaseSymptom) -> None:
    link.delete()

