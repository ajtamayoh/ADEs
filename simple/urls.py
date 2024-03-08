from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('acerca_de_simple', views.acerca_de_simple, name="acerca_de_simple"),
    path('historias_clinicas', views.historias_clinicas, name="historias_clinicas"),
    path('creditos', views.creditos, name="creditos"),
    path('agradecimientos', views.agradecimientos, name="agradecimientos")
]
