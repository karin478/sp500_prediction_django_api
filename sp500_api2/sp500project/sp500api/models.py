
from django.db import models

class SP500Prediction(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    predictions = models.JSONField()

