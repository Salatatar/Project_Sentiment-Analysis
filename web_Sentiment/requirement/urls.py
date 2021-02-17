from django.urls import path
from requirement import views

urlpatterns = [
    path('', views.requirement, name='requirement'),
    path('addRequirement/', views.submitRequire, name='submitRequire'),
]
