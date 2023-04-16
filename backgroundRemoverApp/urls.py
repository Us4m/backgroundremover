from django.urls import path
from . import views

urlpatterns = [
    path('api/remove-background/', views.remove_background, name='remove_background'),
]
