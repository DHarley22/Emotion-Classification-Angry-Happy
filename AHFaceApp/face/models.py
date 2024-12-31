from django.db import models
from django.utils import timezone

class Photo(models.Model):
    face = models.ImageField(upload_to='photos/')  # Sauvegarde l'image dans le dossier 'photos/'
    uploaded_at = models.DateTimeField(default=timezone.now)  # Date et heure de l'upload, avec la valeur par défaut timezone.now


class FaceDetection(models.Model):
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='faces')
    x = models.IntegerField()
    y = models.IntegerField()
    width = models.IntegerField()
    height = models.IntegerField()
