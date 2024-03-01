from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from . import views

urlpatterns = [
    # path('', views.home, name='home'),
    path('set_user/', views.set_user, name='set_user'),
    path('morph/', views.morph_view, name='morph'),
    path('send_email/', views.send_email, name='send_email'),
    path('delete_morphs/', views.delete_morphs, name='delete_morphs'),
    path('get_dataset/', views.get_dataset, name='get_dataset'),
    path('policy/', views.policy, name='policy'),
    ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)