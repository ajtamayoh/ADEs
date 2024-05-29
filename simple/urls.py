from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ades', views.ades, name="ades"),
    path('diseases', views.diseases, name="diseases"),
    path('symptoms2diagnosis', views.symptoms2diagnosis, name="symptoms2diagnosis"),
    path('sonatafyassistant', views.sonatafyassistant, name="sonatafyassistant"),
    path('summarization', views.summarization, name="summarization"),
    path('sentiment-analysis', views.sentiment_analysis, name="sentiment-analysis"),
    path('fake_news_detection', views.fake_news_detection, name="fake_news_detection"),
    path('about_ades', views.about_ades, name="about_ades"),
    path('historias_clinicas', views.historias_clinicas, name="historias_clinicas"),
    path('credits', views.credits, name="credits"),
    path('agradecimientos', views.agradecimientos, name="agradecimientos")
]
