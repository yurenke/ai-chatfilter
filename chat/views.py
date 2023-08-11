from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.http import HttpResponse
import json

js_file = ''
with open('chat/js/axios.min.js', 'r') as f:
    js_file = f.read()

def index(request):
    return render(request, 'index.html', {})

def room(request, room_name = ''):
    return render(request, 'room.html', {
        'room_name_json': mark_safe(json.dumps(room_name))
    })

def controller_zh(request):
    return render(request, 'controller_zh.html', {
        
    })

def controller_en(request):
    return render(request, 'controller_en.html', {
        
    })

def axiosjs(request):
    response = HttpResponse(content=js_file, content_type="application/javascript")
    return response