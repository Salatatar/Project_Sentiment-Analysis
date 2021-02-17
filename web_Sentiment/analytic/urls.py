from django.urls import path
from analytic import views

urlpatterns = [
    path('', views.analytic, name='analytic'),
    # path('', views.analytic_home, name='home_anlysis')
]