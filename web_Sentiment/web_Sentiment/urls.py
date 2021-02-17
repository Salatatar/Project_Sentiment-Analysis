from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('requirement.urls')),
    path('retrieval/', include('retrieval.urls')),
    path('extraction/', include('extraction.urls')),
    path('analytic/', include('analytic.urls')),
    path('contact/', include('contact.urls')),
    path('comparison/', include('comparison.urls')),
    path('financial/', include('financial.urls')),
]
