from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class MLModel(models.Model):
    """stores ml model info"""
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    file_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} v{self.version}"

class MangoImage(models.Model):
    #stores uploaded mango pics and ai predictions
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='mango_images/')
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)  
    
    # ai prediction stuff
    predicted_class = models.CharField(max_length=50, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    disease_type = models.CharField(max_length=20, blank=True) 
    model_used = models.CharField(max_length=20, blank=True) 
    model_filename = models.CharField(max_length=100, blank=True) 
    
    #dashboard stuff
    disease_classification = models.CharField(max_length=50, blank=True)
    is_verified = models.BooleanField(default=False)
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='verified_images')
    verified_date = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    user_feedback = models.TextField(null=True, blank=True)  
    user_confirmed_correct = models.BooleanField(null=True, blank=True)  
    
    #symptoms user picked
    selected_symptoms = models.JSONField(null=True, blank=True)  
    primary_symptoms = models.JSONField(null=True, blank=True)  
    alternative_symptoms = models.JSONField(null=True, blank=True)  
    detected_disease = models.CharField(max_length=50, blank=True)  
    top_diseases = models.JSONField(null=True, blank=True)  
    symptoms_data = models.JSONField(null=True, blank=True)  
    
    #extra info
    image_size = models.CharField(max_length=20, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    
    #gps stuff
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_accuracy = models.FloatField(null=True, blank=True)
    location_consent_given = models.BooleanField(default=False)
    location_accuracy_confirmed = models.BooleanField(null=True, blank=True) 
    location_address = models.TextField(blank=True) 
    location_source = models.CharField(max_length=20, blank=True) 
    #training data gate - admin approves a record for retraining
    training_ready = models.BooleanField(default=False)
    training_notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.original_filename} - {self.predicted_class}"
    
    def save(self, *args, **kwargs):
        #auto fill disease_classification from predicted_class
        if self.predicted_class and not self.disease_classification:
            self.disease_classification = self.predicted_class
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        # remove the file from S3 before deleting the DB record
        if self.image:
            self.image.delete(save=False)
        super().delete(*args, **kwargs)

class PredictionLog(models.Model):
    #logs when ai does prediction
    image = models.ForeignKey(MangoImage, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    user_agent = models.TextField(blank=True)
    response_time = models.FloatField(null=True, blank=True)

    #save prediction outputs here
    probabilities = models.JSONField(null=True, blank=True)
    labels = models.JSONField(null=True, blank=True)
    prediction_summary = models.JSONField(null=True, blank=True)
    raw_response = models.JSONField(null=True, blank=True)
    

    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Prediction log for {self.image.original_filename}"

class UserConfirmation(models.Model):
    #when user confirms the prediction is right or wrong
    image = models.OneToOneField(MangoImage, on_delete=models.CASCADE, related_name='user_confirmation')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    predicted_disease = models.CharField(max_length=50)  
    user_feedback = models.TextField(blank=True)  
    confidence_score = models.FloatField(null=True, blank=True)  
    
    #gps stuff
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_accuracy = models.FloatField(null=True, blank=True)
    location_consent_given = models.BooleanField(default=False)
    location_address = models.TextField(blank=True)  
    
    class Meta:
        ordering = ['-id']
    
    def __str__(self):
        return f"Confirmation: {self.predicted_disease} for {self.image.original_filename}"

class UserProfile(models.Model):
    """extra user info like address"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    #address parts
    province = models.CharField(max_length=100, blank=True)
    city = models.CharField(max_length=100, blank=True)
    barangay = models.CharField(max_length=100, blank=True)
    
    #full address 
    address = models.TextField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        
        if self.province or self.city or self.barangay:
            address_parts = []
            if self.barangay:
                address_parts.append(self.barangay)
            if self.city:
                address_parts.append(self.city)
            if self.province:
                address_parts.append(self.province)
            self.address = ', '.join(address_parts)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Profile for {self.user.username}"

class Notification(models.Model):
    #notifs for admin panel
    NOTIFICATION_TYPES = [
        ('image_upload', 'Image Upload'),
        ('system', 'System'),
        ('alert', 'Alert'),
    ]
    
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='image_upload')
    title = models.CharField(max_length=200)
    message = models.TextField()
    related_image = models.ForeignKey(MangoImage, on_delete=models.CASCADE, null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"

class ModelConfig(models.Model):
    """which model file to use for each detection type — swappable via admin"""
    DETECTION_TYPES = [
        ('leaf', 'Leaf Disease Model'),
        ('fruit', 'Fruit Disease Model'),
        ('gate_leaf', 'Gate Leaf Model'),
        ('gate_fruit', 'Gate Fruit Model'),
    ]
    detection_type = models.CharField(max_length=20, choices=DETECTION_TYPES, unique=True)
    model_filename = models.CharField(max_length=255)
    updated_at     = models.DateTimeField(auto_now=True)
    updated_by     = models.ForeignKey(
        'auth.User', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.detection_type}: {self.model_filename}"


class Symptom(models.Model):
    """
    One canonical symptom key per plant part.

    A 'canonical' key is the authoritative snake_case string stored in
    MangoImage.selected_symptoms and used as XGBoost feature column names.

    IMPORTANT — vector_index ordering:
        vector_index=N means this symptom occupies feature column N in the
        XGBoost binary vector.  The value comes from enumerate(LEAF_SYMPTOMS)
        or enumerate(FRUIT_SYMPTOMS) at seed time.  Never reassign these values
        after any model has been trained — doing so invalidates saved .json
        classifier files without any runtime error.

        Symptoms from SYMPTOMS_MAP that are NOT in the vocabulary (e.g. canker_*,
        weevil_*, gall_*, healthy_*) have vector_index=None and is_in_vocabulary=False.
    """

    PLANT_PART_CHOICES = [
        ('leaf',  'Leaf'),
        ('fruit', 'Fruit'),
    ]

    key = models.SlugField(
        max_length=80,
        help_text="Canonical snake_case key, e.g. 'dark_spots_brown'.",
    )

    plant_part = models.CharField(
        max_length=10,
        choices=PLANT_PART_CHOICES,
        help_text="Which part of the plant this symptom describes.",
    )

    vector_index = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Column index in the XGBoost feature vector. "
            "None for symptoms outside the classifier vocabulary."
        ),
    )

    is_in_vocabulary = models.BooleanField(
        default=False,
        help_text=(
            "True if this key appears in LEAF_SYMPTOMS or FRUIT_SYMPTOMS "
            "and is a valid XGBoost feature. False for display-only keys."
        ),
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('key', 'plant_part')]
        ordering = ['plant_part', 'vector_index', 'key']
        verbose_name = 'Symptom'
        verbose_name_plural = 'Symptoms'

    def __str__(self) -> str:
        idx = f"[{self.vector_index}]" if self.vector_index is not None else "[—]"
        return f"{self.plant_part}:{self.key} {idx}"
    
class SymptomAlias(models.Model):
    """
    Maps a variant or legacy string to one canonical Symptom row.

    Example:  alias='acervuli'  →  canonical=Symptom(key='black_specks_in_lesion', plant_part='leaf')

    Aliases are looked up by the normalize_symptom() utility before indexing
    into the XGBoost feature vector.  They are also useful for accepting
    informal strings from older mobile app versions.
    """
    alias = models.SlugField(
        max_length=120,
        unique=True,
        help_text="The variant / legacy string that will be resolved.",
    )
    canonical = models.ForeignKey(
        Symptom,
        on_delete=models.CASCADE,
        related_name='aliases',
        help_text="The canonical symptom this alias resolves to.",
    )
    source = models.CharField(
        max_length=200,
        blank = True,
        help_text= "Helps auditors trace where "
    )

    class Meta:
        ordering = ['alias']
        verbose_name = 'Symptom Alias'
        verbose_name_plural = 'Symptom Aliases'

    def __str__(self) -> str:
        return f"'{self.alias}' → {self.canonical}"
    
class Disease(models.Model):    
    """
    A disease class that the system can recognise or display.

    is_in_classifier=True  →  the disease appears in LEAF_DISEASES or FRUIT_DISEASES
                               and is a valid output label for the XGBoost and CNN models.
    is_in_classifier=False →  future/display-only diseases such as Bacterial Canker,
                               Cutting Weevil, Gall Midge that have SYMPTOMS_MAP entries
                               but are not yet in the trained classifier.
    """
    PLANT_PART_CHOICES = [
        ('leaf',  'Leaf'),
        ('fruit', 'Fruit'),
    ]

    name = models.CharField(
        max_length=100,
        help_text="readable disease name matching LEAF_CLASS_NAMES / FRUIT_CLASS_NAMES.",
    )

    plant_part = models.CharField(
        max_length=10,
        choices=PLANT_PART_CHOICES,
    )

    is_in_classifier = models.BooleanField(
        default=False,
        help_text=(
            "True if this disease is a valid prediction label in the current "
            "trained XGBoost / CNN models."
        ),
    )

    class Meta:
        unique_together = [('name', 'plant_part')]
        ordering = ['plant_part', 'name']
        verbose_name = 'Disease'
        verbose_name_plural = 'Diseases'

    def __str__(self) -> str:
        flag = "class found" if self.is_in_classifier else "is not in class"
        return f"{self.plant_part}:{self.name} [{flag}]"
    
class DiseaseSymptom(models.Model):
    """
    Through-table connecting a Disease to a Symptom with UI display metadata.

    This replaces the SYMPTOMS_MAP dict in symptom_views.py.

    display_label  — The verbose, human-friendly string shown in the mobile app
                     checklist, e.g. "Irregular brown or black spots on leaves".
                     The same Symptom key (e.g. black_sooty_coating) can have
                     different display_label values for different diseases because
                     the clinical significance differs.

    display_order  — Zero-based sort position within this disease's symptom list.
                     The mobile app renders symptoms in this order, not alphabetical.
    """
    disease = models.ForeignKey(
        Disease,
        on_delete = models.CASCADE,
        related_name = 'disease_symproms',
    )

    symptom = models.ForeignKey(
        Symptom,
        on_delete=models.PROTECT, 
        related_name='disease_symptoms',
    )

    display_label = models.CharField(
        max_length=300,
        help_text="Verbose UI string shown to the user in the mobile symptom checklist.",
    )

    display_order = models.PositiveSmallIntegerField(
        default=0,
        help_text="Sort order within the disease's symptom list (0-based).",
    )

    class Meta:
        # One disease cannot have the same symptom at two different positions.
        unique_together = [('disease', 'display_order')]
        ordering = ['disease', 'display_order']
        verbose_name = 'Disease–Symptom Link'
        verbose_name_plural = 'Disease–Symptom Links'
    def __str__(self) -> str:
        return f"{self.disease.name} ({self.disease.plant_part}) #{self.display_order} → {self.symptom.key}"