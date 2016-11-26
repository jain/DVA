from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^(?P<request_text>[A-Za-z0-9_+-.,!@#$%^&*();|<>"\' ]+)', views.results, name='results'),
]
