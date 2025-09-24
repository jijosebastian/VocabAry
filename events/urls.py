from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detection, name='detection'),
    path('list/', views.list, name='list'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_detected_words/', views.get_detected_words, name='get_detected_words'),
    path('get_objects/', views.get_detected_objects, name='get_detected_objects'),
    
    path('update_score_and_learn/', views.update_score_and_learn, name='update_score_and_learn'),
    path('translate/', views.translate_word, name='translate_word'),


]