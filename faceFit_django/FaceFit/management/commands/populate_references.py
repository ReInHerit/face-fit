import os
import json

from django.conf import settings
from django.core.management.base import BaseCommand
from FaceFit.models import Reference
from django.templatetags.static import static

class Command(BaseCommand):
    help = 'Populate the Reference model with data from image files'

    def handle(self, *args, **options):
        json_file_path = os.path.join(settings.BASE_DIR, 'FaceFit', 'static', 'assets', 'json', 'painting_data.json')
        images_folder = os.path.join(settings.BASE_DIR, 'FaceFit', 'static', 'assets', 'images')  # Path to your images folder

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            for image_filename, image_data in data.items():
                image_path = os.path.join(images_folder, image_filename)
                print(image_path)
                if os.path.exists(image_path):
                    reference = Reference.objects.create(
                        reference_title=image_filename,
                        reference_text=image_data['description'],
                        source=f'/{os.path.join("static", "assets", "images", image_filename)}',  # Construct the correct URL using a leading slash
                        # Add other fields as needed
                    )
                    # reference.save()

                    self.stdout.write(self.style.SUCCESS(f'Successfully created Reference: {reference.reference_title}'))
                else:
                    self.stdout.write(self.style.ERROR(f'Image not found for: {image_filename}'))
