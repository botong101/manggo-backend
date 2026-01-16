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
            default=r'C:\Users\kenta\Downloads\mangosense-main\datasets\split-mango\train',
            help='Source directory containing class folders'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of images per class'
        )

    def handle(self, *args, **options):
        source_dir = options['source_dir']
        limit = options['limit']
        
        # Get or create a default user
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

        # Create media directory if it doesn't exist
        media_root = os.path.join(settings.BASE_DIR, 'media', 'mango_images')
        os.makedirs(media_root, exist_ok=True)

        total_imported = 0
        
        # Process each class folder
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
                
            self.stdout.write(f"Processing class: {class_name}")
            
            # Get all image files in this class
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Apply limit if specified
            if limit:
                image_files = image_files[:limit]
            
            imported_count = 0
            
            for image_file in image_files:
                try:
                    source_path = os.path.join(class_path, image_file)
                    
                    # Skip if already exists
                    if MangoImage.objects.filter(original_filename=image_file).exists():
                        continue
                    
                    # Create unique filename
                    name, ext = os.path.splitext(image_file)
                    unique_filename = f"{class_name}_{name}_{total_imported}{ext}"
                    
                    # Copy image to media directory
                    dest_path = os.path.join(media_root, unique_filename)
                    shutil.copy2(source_path, dest_path)
                    
                    # Get image dimensions
                    with Image.open(dest_path) as img:
                        width, height = img.size
                        image_size = f"{width}x{height}"
                    
                    # Determine disease type (leaf/fruit) - you may need to adjust this logic
                    disease_type = 'leaf' if 'leaf' in image_file.lower() else 'fruit'
                    
                    # Create database record
                    mango_image = MangoImage.objects.create(
                        user=user,
                        image=f'mango_images/{unique_filename}',
                        original_filename=image_file,
                        predicted_class=class_name,
                        confidence_score=random.uniform(0.7, 0.99),  # Random confidence for demo
                        disease_type=disease_type,
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