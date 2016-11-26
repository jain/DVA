from __future__ import unicode_literals

from django.apps import AppConfig
from sklearn.externals import joblib

clf = joblib.load("svm_classifier.pkl")

class DvaAppConfig(AppConfig):
    name = 'dva_app'
