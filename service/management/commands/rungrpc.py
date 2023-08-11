from django.core.management.base import BaseCommand
from grpcservice.apps import GrpcService 


class Command(BaseCommand):
    help = 'use django manage to open grpc service.'


    def add_arguments(self, parser):
        parser.add_argument(
            '-port', dest='port', required=False, help='port of opening connection.',
        )
        # parser.add_argument(
        #     '-host', dest='host', required=False, help='host ip or domain for connection.',
        # )
        parser.add_argument(
            '-webhost', dest='webhost', required=False, help='host ip or domain for connection of websocket (django).',
        )
        parser.add_argument(
            '-webport', dest='webport', required=False, help='port for connection of websocket (django).',
        )


    def handle(self, *args, **options):
        port = options.get('port')
        host = options.get('host')
        webhost = options.get('webhost')
        webport = options.get('webport')
        
        if port is None:
            port = 50051
        else:
            port = int(port)
        
        if host is None:
            host = '0.0.0.0'

        if webhost is None:
            webhost = '127.0.0.1'

        if webport is None:
            webport = 80
        else:
            webport = int(webport)

        try:
            main = GrpcService(host=host, port=port, webhost=webhost, webport=webport)
        except BaseException as err:
            print(err)

