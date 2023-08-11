from django.core.management.base import BaseCommand
from dataparser.apps import ExcelParser, MessageParser
from ai.train import get_row_list_by_json_path
from ai.models import TextbookSentense
import os, json

class Command(BaseCommand):
    help = 'trim all exist text then export json and excel file'

    def add_arguments(self, parser):
        # parser.add_argument(
        #     '-i', dest='input_excel_path', required=True,
        #     help='the path of excel file.',
        # )
        # parser.add_argument(
        #     '-o', dest='output_path', required=False,
        #     help='the path of output data.',
        # )
        pass

    def handle(self, *args, **options):
        path_cwd = os.getcwd()
        json_file_path = os.path.join(path_cwd, 'ai', 'assets', 'textbook', 'json', '2022-02-17.json')
        path_output_excel = os.path.join(path_cwd, '__trimtor_output__.xlsx')
        path_output_json = os.path.join(path_cwd, '__trimtor_output__.json')

        json_results = get_row_list_by_json_path(json_file_path)
        _mp = MessageParser()
        _ep = ExcelParser()


        print('json_results length: ', len(json_results))
        # print('json_results[10]: ', json_results[-100:])
        _result_list = []
        for res in json_results:
            _text, _lv, _anchor = _mp.parse(res[2])
            _status = int(res[3])
            _result_list.append([_text, _status])

        
        db_dataset = TextbookSentense.objects.values_list('id', 'origin', 'text', 'status', 'weight').all()
        for data in db_dataset:
            _text, _lv, _anchor = _mp.parse(data[2])
            _status = int(data[3])
            _result_list.append([_text, _status])
            
        print('_result_list length: ', len(_result_list))
        print('_result_list: ', _result_list[:10])
        print('_result_list last: ', _result_list[-10:])

        # _ep.export_excel(file=path_output_excel, data=_result_list)
        with open(path_output_json, 'w+', encoding='utf-8') as f:
            json_string = json.dumps(_result_list, ensure_ascii=False, indent=2)
            f.write(json_string)
            
        

        