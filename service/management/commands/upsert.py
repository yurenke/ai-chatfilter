from django.core.management.base import BaseCommand
from django.apps import apps

from service.excelhandler import insert_textbook

class Command(BaseCommand):
    help = 'insert or update some data into database'

    def add_arguments(self, parser):
        parser.add_argument(
            '-i', dest='input_excel_path', required=False, help='the path of excel file.',
        )
        parser.add_argument(
            '-model', dest='model', required=False, help='model name',
        )
        

    def handle(self, *args, **options):
        file_path = options.get('input_excel_path', None)
        model_name = options.get('model')

        if model_name == 'textbook':

            insert_textbook(file_path)

        
        