# Generated by Django 4.1 on 2023-10-31 16:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('FaceFit', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='reference',
            name='pub_date',
        ),
    ]
