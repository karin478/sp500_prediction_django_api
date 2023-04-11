from django.urls import path
from .views import SP500PricePredictionView

urlpatterns = [
    path('prediction/', SP500PricePredictionView.as_view(), name='sp500_api'),
]
