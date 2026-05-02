from __future__ import annotations

def get_information_for_disease(disease_name: str, model_used: str) -> str:
    if not disease_name:
        return "No treatment information available - disease name is empty."

    try:
        from mangosense.models import Disease as DiseaseModel
        norm = disease_name.replace('_', ' ').replace('-', ' ').strip().lower()
        for disease in DiseaseModel.objects.filter(plant_part=model_used.lower()):
            if disease.name.replace('_', ' ').replace('-', ' ').strip().lower() == norm:
                if disease.description:
                    return disease.description
                break
    except Exception:
        pass
    
    return f"No information available for '{disease_name}'. Please consult with an agricultural expert."

def get_treatment_for_disease(disease_name: str, model_used: str) -> str:
    if not disease_name:
        return "No treatment information available - disease name is empty."

    try:
        from mangosense.models import Disease as DiseaseModel
        norm = disease_name.replace('_', ' ').replace('-', ' ').strip().lower()
        for disease in DiseaseModel.objects.filter(plant_part=model_used.lower()):
            if disease.name.replace('_', ' ').replace('-', ' ').strip().lower() == norm:
                if disease.treatment:
                    return disease.treatment
                break
    except Exception:
        pass

    # treatment = treatment_suggestions.get(disease_name)
    # if treatment:
    #     return treatment

    # disease_lower = disease_name.lower()
    # for key, value in treatment_suggestions.items():
    #     if key.lower() == disease_lower:
    #         return value

    # disease_normalized = disease_name.replace('_', ' ').replace('-', ' ').strip()
    # for key, value in treatment_suggestions.items():
    #     if disease_normalized.lower() == key.replace('_', ' ').replace('-', ' ').strip().lower():
    #         return value

    return f"No treatment information available for '{disease_name}'. Please consult with an agricultural expert."
