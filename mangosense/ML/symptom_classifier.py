import math 
import json    
import time                        
import threading                   
from pathlib import Path          
from collections import defaultdict, Counter
from typing import Iterable

from mangosense.repositories.symptom_repository import (
    get_vocabulary, get_diseases, normalize_symptom,
)
from mangosense.models import Symptom as SymptomModel 

class SymptomEncoder:
    def __init__(self, plant_part: str):
        vocabulary_symptom_rows = (
            SymptomModel.objects
            .filter(plant_part=plant_part, is_in_vocabulary=True)
            .exclude(vector_index=None)          # guard: skip rows missing a slot
            .values('key', 'vector_index')
            .order_by('vector_index')  
        )

        self.symptom_to_slot: dict[str, int] = {
            symptom_row['key']: symptom_row['vector_index']
            for symptom_row in vocabulary_symptom_rows
        }

        self.feature_vector_length: int = (
            max(self.symptom_to_slot.values()) + 1 if self.symptom_to_slot else 0
        )

        self.vocabulary: list[str] = get_vocabulary(plant_part)

    def encode(self, raw_symptoms: Iterable[str]) -> list[int]:

        feature_vector = [0] * self.feature_vector_length

        for raw_symptom_text in raw_symptoms or ():
            canonical_symptom_key = normalize_symptom(raw_symptom_text)
            feature_slot_number = self.symptom_to_slot.get(canonical_symptom_key)

            if feature_slot_number is not None:
                feature_vector[feature_slot_number] = 1
        
        return feature_vector