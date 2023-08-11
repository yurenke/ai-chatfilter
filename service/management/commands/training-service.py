from django.core.management.base import BaseCommand
from trainingservice.main import LaunchTrainingService
# from configparser import RawConfigParser
# from dataparser.apps import ExcelParser

DEFAULT_TRAINING_SERVICE_PROT = 8031
DEFAULT_REMOTE_WEBSOCKET_PORT = 8000

class Command(BaseCommand):
    help = 'train models'
    client = None
    bufsize = 1024 
    spend_recv_second = 0
    length_right = 0
    length_timeout_no_recv = 0

    def add_arguments(self, parser):
        parser.add_argument(
            '-port', dest='port', required=False, help='port',
        )
        parser.add_argument(
            '-host', dest='host', required=False, help='host',
        )
        parser.add_argument(
            '-webport', dest='webport', required=False, help='websockets port.',
        )
        parser.add_argument(
            '-webhost', dest='webhost', required=False, help='websockets host.',
        )


    def handle(self, *args, **options):
        port = options.get('port')
        host = options.get('host')
        webport = options.get('webport')
        webhost = options.get('webhost')

        if port is None:
            port = DEFAULT_TRAINING_SERVICE_PROT
        else:
            port = int(port)
            
        if host is None:
            host = '0.0.0.0'

        if webport is None:
            webport = DEFAULT_REMOTE_WEBSOCKET_PORT
        else:
            webport = int(webport)

        if webhost is None:
            webhost = '127.0.0.1'

        main = LaunchTrainingService(host, port, webhost, webport)

