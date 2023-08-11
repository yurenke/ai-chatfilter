from django.core.management.base import BaseCommand
from tcpsocket.main import LaunchTcpSocket 


class Command(BaseCommand):
    help = 'use django manage to open tpc socket for chat center packages.'


    def add_arguments(self, parser):
        parser.add_argument(
            '-port', dest='port', required=False, help='port of opening to recive socket connection.',
        )
        parser.add_argument(
            '-host', dest='host', required=False, help='host ip or domain for socket connection.',
        )
        parser.add_argument(
            '-webport', dest='webport', required=False, help='port of connecting to backend websocket.',
        )
        parser.add_argument(
            '-webhost', dest='webhost', required=False, help='ip or domain for connecting to backend websocket.',
        )
        # parser.add_argument(
        #     '-lang', dest='language', required=False, help='language between CH and EN. default is CH.',
        # )


    def handle(self, *args, **options):
        port = options.get('port')
        host = options.get('host')
        webport = options.get('webport')
        webhost = options.get('webhost')
        # language = options.get('language')
        
        if port is None:
            port = 8025
        else:
            port = int(port)
        
        if host is None:
            host = '0.0.0.0'
        
        if webport is None:
            webport = 8000
        else:
            webport = int(webport)

        if webhost is None:
            webhost = '127.0.0.1'
        # bufsize = 1024

        main = LaunchTcpSocket(host, port, webhost, webport)

