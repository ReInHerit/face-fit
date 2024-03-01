import os
from django import forms
from django.db import models
from django.contrib import admin
from django.core.exceptions import ValidationError
from django.http import HttpResponseRedirect
from django.urls import reverse
from .models import Reference
from .forms import ReferenceAdminForm


class ReferenceAdmin(admin.ModelAdmin):
    form = ReferenceAdminForm
    change_form_template = 'admin/custom_change_form.html'

    # class Media:
    #     js = ('assets/js/check_filename.js', )
    #
    # def get_form(self, request, obj=None, **kwargs):
    #     # This method is called to get the form for both Add and Change views
    #     # Append the JavaScript file only for these views
    #     if request.path.endswith('/add/') or '/change/' in request.path:
    #         return super().get_form(request, obj, **kwargs)
    #
    # def get_changeform_initial_data(self, request):
    #     initial = super().get_changeform_initial_data(request)
    #     return initial
    #
    # def get_addform_initial_data(self, request):
    #     initial = super().get_addform_initial_data(request)
    #     return initial
    #
    # def change_view(self, request, object_id, form_url='', extra_context=None):
    #     try:
    #         return super().change_view(request, object_id, form_url, extra_context)
    #     except ValidationError as e:
    #         self.message_user(request, str(e), level='ERROR')
    #         return HttpResponseRedirect(reverse('admin:FaceFit_reference_changelist'))
    #
    # def add_view(self, request, form_url='', extra_context=None):
    #     return super().add_view(request, form_url, extra_context)



# Register your models here.
admin.site.register(Reference, ReferenceAdmin)
