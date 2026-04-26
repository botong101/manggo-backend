from django.apps import AppConfig


class MangosenseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'mangosense'

    def ready(self) -> None:
        from django.db.models.signals import post_delete, post_save

        from mangosense.models import Disease, DiseaseSymptom, Symptom, SymptomAlias
        from mangosense.repositories.symptom_repository import invalidate_symptom_cache

        def _bust_cache(sender, **kwargs): 
            """Signal receiver — busts the symptom cache on any model change."""
            invalidate_symptom_cache()

        # Connect _bust_cache to every relevant model's post_save and post_delete.
        # dispatch_uid prevents duplicate connections if ready() is somehow called
        # more than once (defensive, normally not needed).
        for model in (Symptom, SymptomAlias, Disease, DiseaseSymptom):
            post_save.connect(
                _bust_cache,
                sender=model,
                dispatch_uid=f'bust_symptom_cache_save_{model.__name__}',
            )
            post_delete.connect(
                _bust_cache,
                sender=model,
                dispatch_uid=f'bust_symptom_cache_delete_{model.__name__}',
            )