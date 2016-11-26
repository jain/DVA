from django.http import HttpResponse
from apps import clf
def index(request):
    return HttpResponse("You're at the DVA index.")

def results(request,request_text):
    l = []
    l.append(request_text)
    prediction = clf.predict(l)
    response = "%s"
    return HttpResponse(response % prediction[0])
