from django.urls import path
from extraction import views

urlpatterns = [
    path('', views.extraction, name='extraction'),
]