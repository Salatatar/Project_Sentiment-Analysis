from django.urls import path
from financial import views

urlpatterns = [
    path('', views.financial, name='financial'),
    path('financial-info', views.financial_info, name='financial_info'),
    path('condition', views.condition, name='condition'),
    path('result', views.result, name='result'),
]