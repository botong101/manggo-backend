from rest_framework import serializers
from django.contrib.auth.models import User
from .models import MangoImage, MLModel, PredictionLog, UserProfile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']

class UserDetailSerializer(serializers.ModelSerializer):
    profile = serializers.SerializerMethodField()
    total_images = serializers.SerializerMethodField()
    verified_images = serializers.SerializerMethodField()
    full_name = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'full_name', 'date_joined', 'is_active', 'profile', 'total_images', 'verified_images']
        read_only_fields = ['id', 'date_joined']
    
    def get_profile(self, obj):
        try:
            profile = obj.userprofile
            return {
                'province': profile.province,
                'city': profile.city,
                'barangay': profile.barangay,
                'postal_code': profile.postal_code,
                'address': profile.address,
                'phone': profile.phone,
                'full_address': f"{profile.barangay}, {profile.city}, {profile.province} {profile.postal_code}".strip(', ') if profile.barangay else profile.address
            }
        except UserProfile.DoesNotExist:
            return None
    
    def get_total_images(self, obj):
        return obj.mangoimage_set.count()
    
    def get_verified_images(self, obj):
        return obj.mangoimage_set.filter(is_verified=True).count()
    
    def get_full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}".strip() or obj.username

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    full_address = serializers.SerializerMethodField()
    
    class Meta:
        model = UserProfile
        fields = ['user', 'province', 'city', 'barangay', 'postal_code', 'address', 'full_address', 'phone', 'created_at']
        read_only_fields = ['created_at', 'full_address']
    
    def get_full_address(self, obj):
        """put address parts together"""
        address_parts = []
        if obj.barangay:
            address_parts.append(obj.barangay)
        if obj.city:
            address_parts.append(obj.city)
        if obj.province:
            address_parts.append(obj.province)
        if obj.postal_code:
            address_parts.append(obj.postal_code)
        return ', '.join(address_parts) if address_parts else obj.address

class MangoImageSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    user = UserSerializer(read_only=True)

    class Meta:
        model = MangoImage
        fields = [
            'id', 'user', 'image', 'image_url', 'original_filename', 'predicted_class',
            'confidence_score', 'uploaded_at', 'is_verified', 'notes', 'disease_classification',
            'verified_by', 'verified_date', 'disease_type', 'user_feedback', 'user_confirmed_correct',
            'latitude', 'longitude', 'location_address', 'location_source', 'location_consent_given', 'location_accuracy_confirmed',
            #symptoms stuff
            'selected_symptoms', 'primary_symptoms', 'alternative_symptoms', 'detected_disease', 'top_diseases', 'symptoms_data'
        ]
        read_only_fields = [
            'id', 'uploaded_at', 'predicted_class', 'confidence_score',
            'disease_classification', 'image_size', 'processing_time', 'user_confirmed_correct'
        ]

    def get_image_url(self, obj):
        request = self.context.get('request')
        if request is not None:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url if obj.image else None

class MangoImageUpdateSerializer(serializers.ModelSerializer):
    """for updating mango image records"""
    class Meta:
        model = MangoImage
        fields = [
            'predicted_class', 'confidence_score', 'disease_type',
            'image_size', 'processing_time', 'is_verified'
        ]
        #fields that can be updated
        extra_kwargs = {
            'predicted_class': {'required': False},
            'confidence_score': {'required': False},
            'disease_type': {'required': False},
            'image_size': {'required': False},
            'processing_time': {'required': False},
            'is_verified': {'required': False},
        }

class BulkUpdateSerializer(serializers.Serializer):
    """for updating lots of images at once"""
    image_ids = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=1,
        help_text="List of image IDs to update"
    )
    updates = serializers.DictField(
        help_text="Dictionary of fields to update"
    )
    
    def validate_image_ids(self, value):
        """make sure images exist"""
        existing_ids = MangoImage.objects.filter(id__in=value).values_list('id', flat=True)
        missing_ids = set(value) - set(existing_ids)
        if missing_ids:
            raise serializers.ValidationError(f"Images with IDs {list(missing_ids)} do not exist")
        return value
    
    def validate_updates(self, value):
        """check update fields are ok"""
        allowed_fields = [
            'predicted_class', 'confidence_score', 'disease_type',
            'image_size', 'processing_time', 'is_verified'
        ]
        
        invalid_fields = set(value.keys()) - set(allowed_fields)
        if invalid_fields:
            raise serializers.ValidationError(f"Invalid fields: {list(invalid_fields)}")
        
        return value

class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = ['id', 'name', 'version', 'file_path', 'created_at', 'is_active']
        read_only_fields = ['id', 'created_at']

class PredictionLogSerializer(serializers.ModelSerializer):
    image = MangoImageSerializer(read_only=True)
    
    class Meta:
        model = PredictionLog
        fields = ['id', 'image', 'timestamp', 'user_agent', 'response_time']
        read_only_fields = ['id', 'timestamp']

class ImageUploadSerializer(serializers.Serializer):
    """for uploading images"""
    image = serializers.ImageField()
    
    def validate_image(self, value):
        #max 5mb
        if value.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("Image size cannot exceed 5MB")
        
        #only these formats ok:
        allowed_formats = ['JPEG', 'JPG', 'PNG']
        if value.image.format not in allowed_formats:
            raise serializers.ValidationError("Only JPEG and PNG images are allowed")
        
        return value