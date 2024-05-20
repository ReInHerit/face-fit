import os
from django import forms
from django.db import models
from django.contrib import admin
from django.contrib.admin.actions import delete_selected as delete_selected_original
from django.core.exceptions import ValidationError
from django.http import HttpResponseRedirect
from django.urls import reverse
from .models import Reference
from .forms import ReferenceAdminForm

def delete_selected(modeladmin, request, queryset):
    for obj in queryset:
        obj.delete()

delete_selected.short_description = delete_selected_original.short_description
class ReferenceAdmin(admin.ModelAdmin):
    form = ReferenceAdminForm
    change_form_template = 'admin/custom_change_form.html'
    actions = [delete_selected]


# Register your models here.
admin.site.register(Reference, ReferenceAdmin)
