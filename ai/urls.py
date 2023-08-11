from django.urls import path
from django.http import Http404, HttpResponse, JsonResponse
# from ai.grpc import account_pb2_grpc

def defaultoutput(request):
    return JsonResponse({'ai': '---'})


urlpatterns = [
    path('', defaultoutput, name='index'),
]