import os

from django import forms
from django.core.exceptions import ValidationError

from .models import Reference


class ReferenceAdminForm(forms.ModelForm):
    class Meta:
        model = Reference
        fields = ['reference_title', 'reference_text', 'source']
        search_fields = ['source']
        widgets = {'reference_text': forms.Textarea(attrs={'rows': 10, 'cols': 80})}

    def __init__(self, *args, **kwargs):
        super(ReferenceAdminForm, self).__init__(*args, **kwargs)
        self.fields['reference_title'].widget.attrs['readonly'] = True

    def set_reference_title_from_image(self, instance):
        # Run your logic to set reference_title based on the selected image
        # You can use get_upload_to logic here
        # For demonstration purposes, assuming the image name is stored in instance.source.name
        instance.reference_title = os.path.splitext(os.path.basename(instance.source.name))[0]

    def save_form_data(self, instance, cleaned_data):
        # Set reference_title from the selected image before saving
        self.set_reference_title_from_image(instance)
        super().save_form_data(instance, cleaned_data)
