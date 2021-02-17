from django.urls import path
from retrieval import views

urlpatterns = [
    path('', views.retrieval, name='retrieval'),
]
