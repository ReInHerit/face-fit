# Generated by Django 4.1 on 2023-12-06 14:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FaceFit', '0008_remove_richtext_parent_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='reference',
            name='source',
            field=models.ImageField(blank=True, null=True, upload_to='media/images/'),
        ),
    ]