from rest_framework import serializers

class SP500PredictionSerializer(serializers.Serializer):
    prediction = serializers.ListField(child=serializers.FloatField())
