# prediction/urls.py

from django.urls import path
from .views import PredictAPIView, RegisterView, LoginView  # Make sure all views are imported

urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),


]
