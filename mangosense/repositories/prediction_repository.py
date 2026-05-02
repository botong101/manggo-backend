from __future__ import annotations

import os
import uuid

from django.conf import settings

from mangosense.models import MangoImage, Notification, PredictionLog


def create_mango_image(
    image_file,
    prediction_summary: dict,
    model_used: str,
    model_path: str,
    original_size: tuple,
    processing_time: float,
    user,
    location_data: dict,
    user_feedback: str,
    is_detection_correct: bool,
    selected_symptoms: list,
    primary_symptoms: list,
    alternative_symptoms: list,
    detected_disease: str,
    top_diseases: list,
    symptoms_data: dict,
) -> MangoImage:
    image_file.seek(0)
    ext = os.path.splitext(image_file.name)[-1].lower() or '.jpg'
    image_file.name = f"tmp_{uuid.uuid4().hex}{ext}"

    return MangoImage.objects.create(
        image=image_file,
        original_filename=image_file.name,
        predicted_class=prediction_summary['primary_prediction']['disease'],
        disease_classification=prediction_summary['primary_prediction']['disease'],
        disease_type=model_used,
        model_used=model_used,
        model_filename=os.path.basename(model_path),
        confidence_score=prediction_summary['primary_prediction']['confidence'] / 100,
        user=user if user and user.is_authenticated else None,
        image_size=f"{original_size[0]}x{original_size[1]}",
        processing_time=processing_time,
        notes=f"Predicted via mobile app with {prediction_summary['primary_prediction']['confidence']:.2f}% confidence",
        is_verified=False,
        user_feedback=user_feedback if user_feedback else None,
        user_confirmed_correct=is_detection_correct if user_feedback else None,
        selected_symptoms=selected_symptoms if selected_symptoms else None,
        primary_symptoms=primary_symptoms if primary_symptoms else None,
        alternative_symptoms=alternative_symptoms if alternative_symptoms else None,
        detected_disease=detected_disease if detected_disease else prediction_summary['primary_prediction']['disease'],
        top_diseases=top_diseases if top_diseases else None,
        symptoms_data=symptoms_data if symptoms_data else None,
        **location_data,
    )


def rename_s3_image(mango_image: MangoImage) -> None:
    try:
        import boto3
        disease_slug = mango_image.predicted_class.replace(' ', '_')
        conf_pct = int(mango_image.confidence_score * 100)
        ext_final = os.path.splitext(mango_image.image.name)[-1] or '.jpg'
        old_key = mango_image.image.name
        new_key = f"mango_images/{mango_image.id}_{disease_slug}_{conf_pct}pct{ext_final}"

        s3 = boto3.client(
            's3',
            endpoint_url=settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME,
        )
        s3.copy_object(
            Bucket=settings.AWS_STORAGE_BUCKET_NAME,
            CopySource={'Bucket': settings.AWS_STORAGE_BUCKET_NAME, 'Key': old_key},
            Key=new_key,
        )
        s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=old_key)
        mango_image.image.name = new_key
        mango_image.save(update_fields=['image'])
    except Exception as rename_err:
        print(f"[RENAME WARNING] S3 rename failed, keeping temp name: {rename_err}")


def create_notification(mango_image: MangoImage, model_used: str, prediction_summary: dict) -> None:
    try:
        notification_user = mango_image.user if mango_image.user else None
        if not notification_user:
            from django.contrib.auth.models import User
            notification_user = User.objects.filter(is_staff=True).first()
        if notification_user:
            Notification.objects.create(
                notification_type='image_upload',
                title=f'New {model_used.title()} Image Upload',
                message=(
                    f'A new {model_used} image "{mango_image.original_filename}" was uploaded '
                    f'and classified as {prediction_summary["primary_prediction"]["disease"]} '
                    f'with {prediction_summary["primary_prediction"]["confidence"]:.1f}% confidence.'
                ),
                related_image=mango_image,
                user=notification_user,
            )
        else:
            print("No user available for notification creation")
    except Exception as notification_error:
        print(f"Error creating notification: {notification_error}")


def create_prediction_log(
    mango_image,
    prediction,
    model_class_names: list,
    prediction_summary: dict,
    response_data: dict,
    response_time: float,
    user_agent: str,
) -> None:
    try:
        probs_list = prediction.tolist() if hasattr(prediction, 'tolist') else list(map(float, prediction))
        PredictionLog.objects.create(
            image=mango_image,
            user_agent=user_agent,
            response_time=response_time,
            probabilities=probs_list,
            labels=model_class_names,
            prediction_summary=prediction_summary,
            raw_response=response_data,
        )
    except Exception as log_err:
        print(f"Failed to log prediction activity: {log_err}")
