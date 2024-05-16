from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ades', views.ades, name="ades"),
    path('diseases', views.diseases, name="diseases"),
    path('symptoms2diagnosis', views.symptoms2diagnosis, name="symptoms2diagnosis"),
    path('sonatafyassistant', views.sonatafyassistant, name="sonatafyassistant"),
    path('about_ades', views.about_ades, name="about_ades"),
    path('historias_clinicas', views.historias_clinicas, name="historias_clinicas"),
    path('credits', views.credits, name="credits"),
    path('agradecimientos', views.agradecimientos, name="agradecimientos")
]
