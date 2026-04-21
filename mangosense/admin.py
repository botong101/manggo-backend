from django.contrib import admin
from django.utils.html import format_html
from .models import MangoImage, MLModel, ModelConfig

class TrainingStatusFilter(admin.SimpleListFilter):
    title = "Training Status"
    parameter_name = "training_status"

    def lookups(self, request, model_admin):
        return [
            ("ready", "Ready for Training"),
            ("not_ready", "Not Ready"),
            ("verified_not_approved", "Verified but Not Approved"),
        ]
    
    def queryset(self, request, queryset):
        if self.value() == "ready":
            return queryset.filter(training_ready=True)
        if self.value() == "not_ready":
            return queryset.filter(training_ready=False)
        if self.value() == "verified_not_approved":
            return queryset.filter(is_verified=True, training_ready=False)
        return queryset

@admin.register(MangoImage)
class MangoImageAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "original_filename",
        "predicted_class",
        "disease_type",
        "is_verified",
        "training_ready_display",
        "has_symptoms",
        "uploaded_at",
    ]
    list_filter = [TrainingStatusFilter, "is_verified", "disease_type", "predicted_class"]
    search_fields = ["original_filename", "predicted_class", "disease_classification"]
    readonly_fields = [
        "uploaded_at",
        "predicted_class",
        "confidence_score",
        "disease_type",
        "selected_symptoms",
    ]

    actions = ["mark_training_ready", "mark_training_not_ready", "bulk_verify_and_approve"]

    fieldsets = (
        ("Image Info", {
            "fields": ("original_filename", "image", "uploaded_at", "user"),
        }),
        ("AI Prediction (read-only)", {
            "fields": ("predicted_class", "confidence_score", "disease_type", "disease_classification"),
        }),
        ("Admin Verification", {
            "fields": ("is_verified", "verified_by", "verified_date", "notes"),
        }),
        ("Training Approval", {
            "fields": ("training_ready", "training_notes"),
            "description": (
                "Set training_ready=True only when: disease label is correct, "
                "symptoms are filled in, and data quality is acceptable."
            ),
        }),
        ("Symptom Data", {
            "fields": ("selected_symptoms", "primary_symptoms", "top_diseases"),
            "classes": ("collapse",),
        }),
    )

    def training_ready_display(self, obj):
        if obj.training_ready:
            return format_html('<span style="color:green;font-weight:bold;">Ready</span>')
        return format_html('<span style="color:#999;">Not Ready</span>')
    training_ready_display.short_description = "Training"

    def has_symptoms(self, obj):
        has = bool(obj.selected_symptoms)
        color = "green" if has else "red"
        label = "Yes" if has else "No"
        return format_html(f'<span style="color:{color};">{label}</span>')
    has_symptoms.short_description = "Symptoms"

    @admin.action(description="Mark selected as Training Ready")
    def mark_training_ready(self, request, queryset):
        updated = queryset.update(training_ready=True)
        self.message_user(request, f"{updated} image(s) marked as training ready.")

    @admin.action(description="Mark selected as NOT Training Ready")
    def mark_training_not_ready(self, request, queryset):
        updated = queryset.update(training_ready=False)
        self.message_user(request, f"{updated} image(s) marked as not ready.")

    @admin.action(description="Verify AND approve for training (bulk)")
    def bulk_verify_and_approve(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(
            is_verified=True,
            verified_by=request.user,
            verified_date=timezone.now(),
            training_ready=True,
        )
        self.message_user(request, f"{updated} image(s) verified and approved for training.")


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ["name", "version", "is_active", "created_at"]
    list_filter = ["is_active"]


@admin.register(ModelConfig)
class ModelConfigAdmin(admin.ModelAdmin):
    list_display = ["detection_type", "model_filename", "updated_at", "updated_by"]
