from django.core.management.base import BaseCommand
from django.apps import apps
import os

class Command(BaseCommand):
    help = "clear data in database."

    def add_arguments(self, parser):
        parser.add_argument(
            '-m', dest='model_name', required=True,
            help='the name of model.',
        )
        parser.add_argument(
            '-a', dest='app_name', required=False,
            help='the name of app.',
        )

    def handle(self, *args, **options):
        model_name = options.get('model_name', None)
        app_name = options.get('app_name', 'service')
        if model_name:
            try:
                Model = apps.get_model(app_label=app_name, model_name=model_name)
                print('Model: ', Model)
                Model.objects.all().delete()
                
                print('Clear all data.')
            except Exception as err:
                print(err)


        