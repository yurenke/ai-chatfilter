import logging
from django.core.management.base import BaseCommand
from configparser import RawConfigParser

from service.instance import get_main_service
from dataparser.apps import ExcelParser

import os, sys, time, json

class Command(BaseCommand):
    help = 'test accuracy'

    def add_arguments(self, parser):
        parser.add_argument(
            '-origin', dest='origin_name', required=True, help='origin name.',
        )
        parser.add_argument(
            '-limit', dest='limit', required=False, help='limit size.',
        )


    def handle(self, *args, **options):
        # _handle_start_time = time.time()
        origin_name = options.get('origin_name', None)
        limit = int(options.get('limit', 10000))
        service = get_main_service(is_admin=True)
        service.open_mind()
        result = service.get_test_accuracy_by_origin(origin=origin_name, limit=limit)

        json_file_path = os.getcwd() + '/__testaccuracy__.json'

        with open(json_file_path, 'w+', encoding = 'utf8') as handle:
            json_string = json.dumps(result, ensure_ascii=False, indent=2)
            handle.write(json_string)

        print('done.  acc: ', result['acc'])

