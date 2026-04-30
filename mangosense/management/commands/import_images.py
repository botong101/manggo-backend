import os
import shutil
from django.core.management.base import BaseCommand
from django.conf import settings
from mangosense.models import MangoImage
from django.contrib.auth.models import User
from PIL import Image
import random

class Command(BaseCommand):
    help = 'Import images from training dataset to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--source-dir',
            type=str,
            required=True,
            help='Source directory containing class-named subfolders'
        )
        parser.add_argument(
            '--disease-type',
            type=str,
            choices=['leaf', 'fruit'],
            default='leaf',
            help='Whether images are leaf or fruit (default: leaf)'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of images per class'
        )
        parser.add_argument(
            '--verify',
            action='store_true',
            default=True,
            help='Mark imported images as verified so the retrain pipeline picks them up'
        )

    def handle(self, *args, **options):
        source_dir = options['source_dir']
        limit = options['limit']
        disease_type = options['disease_type']
        is_verified = options['verify']

        user, created = User.objects.get_or_create(
            username='admin',
            defaults={
                'email': 'admin@example.com',
                'is_staff': True,
                'is_superuser': True,
            }
        )
        if created:
            user.set_password('admin123')
            user.save()
            self.stdout.write(f"Created admin user")

        media_root = os.path.join(settings.BASE_DIR, 'media', 'mango_images')
        os.makedirs(media_root, exist_ok=True)

        total_imported = 0

        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            self.stdout.write(f"Processing class: {class_name}")

            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if limit:
                image_files = image_files[:limit]

            imported_count = 0

            for image_file in image_files:
                try:
                    source_path = os.path.join(class_path, image_file)

                    if MangoImage.objects.filter(original_filename=image_file).exists():
                        continue

                    name, ext = os.path.splitext(image_file)
                    unique_filename = f"{class_name}_{name}_{total_imported}{ext}"

                    dest_path = os.path.join(media_root, unique_filename)
                    shutil.copy2(source_path, dest_path)

                    with Image.open(dest_path) as img:
                        width, height = img.size
                        image_size = f"{width}x{height}"

                    MangoImage.objects.create(
                        user=user,
                        image=f'mango_images/{unique_filename}',
                        original_filename=image_file,
                        predicted_class=class_name,
                        disease_classification=class_name,
                        confidence_score=random.uniform(0.7, 0.99),
                        disease_type=disease_type,
                        model_used=disease_type,
                        is_verified=is_verified,
                        image_size=image_size,
                        processing_time=random.uniform(0.1, 0.5)
                    )

                    imported_count += 1
                    total_imported += 1

                    if imported_count % 10 == 0:
                        self.stdout.write(f"  Imported {imported_count} images for {class_name}")

                except Exception as e:
                    self.stdout.write(f"Error processing {image_file}: {str(e)}")

            self.stdout.write(f"Completed {class_name}: {imported_count} images imported")

        self.stdout.write(
            self.style.SUCCESS(f'Successfully imported {total_imported} images total')
        )
