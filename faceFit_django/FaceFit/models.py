import glob
# class Morph(models.Model):
#     morph_name = models.CharField(max_length=200)
#     source = models.CharField(max_length=200)
#     reference = models.ForeignKey(Reference, on_delete=models.CASCADE)
#     def __str__(self):
#         return self.morph_name
import os
import re

from django.core.files.storage import default_storage
from django.db import models
from PIL import Image

from djangoProject import settings


def get_upload_to(instance, filename):
    """
    Generate the upload path for the image.
    If the filename already exists, append a number to make it unique.
    """
    # base_path = os.path.join('media', 'images')
    # file_path = os.path.join(base_path, filename)

    # Construct the absolute file path within the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_path = os.path.join(project_dir, 'static', 'assets', 'images', 'references')
    print('Images path:', images_path)
    image_files = glob.glob(os.path.join(images_path, '*.jpg')) + glob.glob(os.path.join(images_path, '*.png'))
    print('Image files:', image_files)
    # Extract indices from filenames
    existing_indices = set()
    for image_file in image_files:
        match = re.search(r'image(\d+)', os.path.basename(image_file))
        if match:
            index = int(match.group(1))
            existing_indices.add(index)

    # Find gaps in the enumeration
    gaps_indices = []
    last_index = 0  # Initialize last_index to handle the case when there are no existing images
    for index in sorted(existing_indices):
        if index - last_index > 1:
            # There is a gap between last_index and index
            gaps_indices.extend(range(last_index + 1, index))
        last_index = index

    print("Existing Indices:", existing_indices)
    print("Gaps Indices:", gaps_indices)
    images_length = len(image_files)
    absolute_path = os.path.join(images_path, filename)
    # If there are existing image files, find the first available name
    new_filename = filename
    index = 0
    match = re.search(r'image(\d+)', filename)
    if images_length > 0:
        print('images_length:', images_length)
        if match:
            index = int(match.group(1))
            print('index:', index)
            # Check if the index exists in existing_indices
            if index in existing_indices:
                print('index exists', index, existing_indices)
                # If there's a gap, name it with the gap enumeration
                if gaps_indices:
                    print('gap exists', gaps_indices[0], gaps_indices)
                    new_filename = f"image{gaps_indices[0]:02d}.jpg"
                # If there's no gap, name it with the last possible index
                else:
                    print('no gap', max(existing_indices) + 1, existing_indices)
                    new_filename = f"image{max(existing_indices) + 1:02d}.jpg"
            else:
                print('index does not exist', index, 'in', existing_indices)
                # If the index doesn't exist, check if there's a gap
                if gaps_indices:
                    print('gap exists', gaps_indices[0], gaps_indices)
                    # If there's a gap, name it with the gap enumeration
                    new_filename = f"image{gaps_indices[0]:02d}.jpg"
                # If there's no gap, name it with the last possible index
                else:
                    print('no gap', max(existing_indices) + 1, existing_indices)
                    new_filename = f"image{max(existing_indices) + 1:02d}.jpg"
        else:
            print('no match')
            # If the uploaded image doesn't follow the naming rule,
            # name it with the name rule with the digits of the first available gap if one exists;
            # otherwise, name it with the last possible index
            if gaps_indices:
                print('gap exists', gaps_indices[0], gaps_indices)
                new_filename = f"image{min(gaps_indices):02d}.jpg"
            else:
                print('no gap', max(existing_indices) + 1, existing_indices)
                new_filename = f"image{max(existing_indices) + 1:02d}.jpg"
    else:
        print('no images')
        # If there are no existing image files, name it with the name rule
        new_filename = "image01.jpg"
    return new_filename


class Reference(models.Model):
    reference_title = models.CharField(max_length=255)
    reference_text = models.TextField()
    source = models.ImageField(upload_to='assets/images/references', blank=True, null=True)

    def save(self, *args, **kwargs):
        if self.source:
            # Open the uploaded file
            with self.source.open() as uploaded_file:
                # Read the contents of the uploaded file
                contents = uploaded_file.read()

            # Create a new file in the static/assets/images/references directory
            new_file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'images', 'references',
                                         os.path.basename(self.source.name))

            # Ensure the directory exists
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

            # Write the contents of the uploaded file to the new file
            with open(new_file_path, 'wb') as new_file:
                new_file.write(contents)

            # Update the source field to point to the new file
            self.source = os.path.join('assets', 'images', 'references', os.path.basename(self.source.name))

        super().save(*args, **kwargs)
    def delete(self, *args, **kwargs):
        # Construct the absolute file path
        file_path = os.path.join(settings.BASE_DIR, 'static', 'assets', 'images', 'references', os.path.basename(self.source.name))
        print('file path:', file_path)
        # Check if the file exists
        if os.path.isfile(file_path):
            print('is file')
            # Delete the file
            os.remove(file_path)

        # Call the "real" delete() method
        super().delete(*args, **kwargs)
    def __str__(self):
        return self.reference_title

