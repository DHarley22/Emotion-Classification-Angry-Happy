# Generated by Django 5.0.6 on 2024-06-13 21:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face', '0005_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='FaceDetection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('width', models.IntegerField()),
                ('height', models.IntegerField()),
                ('photo', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='faces', to='face.photo')),
            ],
        ),
    ]
