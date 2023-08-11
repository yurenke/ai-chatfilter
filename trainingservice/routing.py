from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/$', consumers.ChatConsumer),
]

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})