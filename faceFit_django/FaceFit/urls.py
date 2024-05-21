from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include, re_path
from django.views.static import serve

from . import views

urlpatterns = [
    # path('', views.home, name='home'),
    path('set_user/', views.set_user, name='set_user'),
    path('morph/', views.morph_view, name='morph'),
    path('send_email/', views.send_email, name='send_email'),
    path('delete_morphs/', views.delete_morphs, name='delete_morphs'),
    path('get_dataset/', views.get_dataset, name='get_dataset'),
    path('policy/', views.policy, name='policy'),
    path('get_dataset_length/', views.get_dataset_length, name='get_dataset_length'),
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
