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
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.original_filename} - {self.predicted_class}"
    
    def save(self, *args, **kwargs):
        #auto fill disease_classification from predicted_class
        if self.predicted_class and not self.disease_classification:
            self.disease_classification = self.predicted_class
        super().save(*args, **kwargs)

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
    DETECTION_TYPES = [
        ('leaf', 'Leaf Model'),
        ('fruit', 'Fruit Model'),
    ]
    detection_type = models.CharField(max_length=10, choices=DETECTION_TYPES, unique=True)
    model_filename = models.CharField(max_length=255)
    updated_at     = models.DateTimeField(auto_now=True)
    updated_by     = models.ForeignKey(
        'auth.User', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.detection_type}: {self.model_filename}"
