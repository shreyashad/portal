from django.urls import path
from .import views

urlpatterns=[
    path('register/', views.register,name='register'),
    path('', views.login, name='login'),
    path('Management_registration/',views.Management_registration.as_view(),name = 'Management_registration')
]