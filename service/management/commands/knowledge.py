from django.core.management.base import BaseCommand
from ai.knowledge.main import KnowledgeCenter
import os

class Command(BaseCommand):
    help = "knowledge, to know how about words"

    def add_arguments(self, parser):
        parser.add_argument('-i', dest='file', type=str, help="the file path of need.")
        parser.add_argument('-di', dest='digital_file', type=str, help="the file path of need.")
        parser.add_argument('-lan', dest='language', type=str, help="the language of this file.", required=False)
        parser.add_argument('-f', dest='freq', type=str, help="freq of the average.", required=False)

    def handle(self, *args, **options):
        _file = options.get('file', '')
        _digital_file = options.get('digital_file', '')
        _language = options.get('language', None)
        _freq = options.get('freq', 1)
        if _freq:
            _freq = int(_freq)
        kc = KnowledgeCenter()

        if _file:

            _full_file_path = os.getcwd() + '/' + _file
            
            if _language:

                kc.absorb_dictionary(_full_file_path, language_code=_language, freq=_freq)
            else:
                kc.absorb_dictionary(_full_file_path, freq=_freq)

        elif _digital_file:

            _full_file_path = os.getcwd() + '/' + _digital_file

            kc.absorb_digital_dictionary(_full_file_path)

        else:

            kc.check_dictionary()
        
    

        