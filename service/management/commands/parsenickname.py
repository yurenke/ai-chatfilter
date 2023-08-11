from django.core.management.base import BaseCommand
from django.apps import apps
from dataparser.apps import ExcelParser, MessageParser
from dataparser.jsonparser import JsonParser
import os, json
from datetime import date

class Command(BaseCommand):
    help = "parse the excel file to json data."

    def add_arguments(self, parser):
        parser.add_argument(
            '-i', dest='input_file', required=True,
            help='the path of excel file.',
        )
        parser.add_argument(
            '-o', dest='output_path', required=False,
            help='the name of ouput file.',
        )

    def handle(self, *args, **options):
        input_file = options.get('input_file')
        output_path = options.get('output_path', None)

        if output_path is None:
            _dirname = os.path.dirname(input_file)
            _filename = '{}.json'.format(date.today())
            output_path = os.path.join(_dirname, _filename)

        try:
            _ep = ExcelParser(file=input_file)
            _basic_model_columns = [['NICKNAME'], ['COUNT(*)'], ['原因']]
            result_list = _ep.get_row_list(column=_basic_model_columns)
            _idx = 0
            _checking_map = {}
            _has_duplicate = False

            output_result = []

            for res in result_list:
                nickname = res[0]
                count = int(res[1])
                reason = res[2]



            print('result_list length: ', len(result_list))
            print('output_path: ', output_path)

            _jp = JsonParser(file=output_path)
            _jp.save(output_result)
            
        except Exception as err:
            print(err)


        