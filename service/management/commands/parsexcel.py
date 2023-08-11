from django.core.management.base import BaseCommand
from django.apps import apps
from dataparser.apps import ExcelParser, MessageParser
from dataparser.jsonparser import JsonParser
from ai.classes.translator_pinyin import translate_by_string
import os, json, re
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
            help='the name of app.',
        )
        parser.add_argument(
            '-c', dest='check_file_path', required=False,
            help='json file path for check.',
        )
        parser.add_argument(
            '-flw', dest='filter_low_weight', required=False,
            help='filter low weight.',
        )
    

    def find_index_of_rlist(self, msg, list):
        _finded = False
        _r_idx = 0
        for _ in list:
            if _[2] == msg:
                _finded = True
                break
            _r_idx += 1
        return _r_idx if _finded else -1

    def handle(self, *args, **options):
        input_file = options.get('input_file')
        output_path = options.get('output_path', None)
        check_file_path = options.get('check_file_path', None)
        filter_low_weight = bool(options.get('filter_low_weight', False))
        
        result_list = []

        if output_path is None:
            _dirname = os.path.dirname(input_file)
            _filename = '{}.json'.format(date.today())
            output_path = os.path.join(_dirname, '../json/',_filename)

        if isinstance(check_file_path, str) and len(check_file_path) > 2:
            _jp = JsonParser(file=check_file_path)
            _old_json = _jp.load()
            _length_json = len(_old_json)
            _j_idx = 0
            _half_length = int(_length_json / 2)
            print('')
            for _oj in _old_json:
                _j_idx += 1
                if _j_idx % 100 ==0:
                    print('Handle Old Json.. [ {:.1%} ]'.format(_j_idx / _length_json), end='\r')
                _length_oj = len(_oj)
                
                status = int(_oj[3])
                if status == 5:
                    continue

                # if _length_oj == 4:
                #     result_list.append(_oj)
                #     continue
                
                msg = _oj[4] if _length_oj>4 else _oj[2]
                weight = int(_oj[1]) if _oj[1] else 1

                if filter_low_weight and _j_idx > 10000 and weight <= 1:
                    continue

                _r_idx = self.find_index_of_rlist(msg, result_list)
                
                if _r_idx >= 0:
                    result_list[_r_idx][1] += weight
                    result_list[_r_idx][3] = status
                else:
                    if weight > 2:
                        weight -= 1
                    elif _j_idx < _half_length and weight == 2:
                        weight = 1
                    result_list.append([_oj[0], weight, msg, status])
                

        try:
            print('Start Handle Excel.')
            _ep = ExcelParser(file=input_file)
            _basic_model_columns = [['VID', '房號'], ['WEIGHT', '權重'], ['MESSAGE', '聊天信息', '禁言内容', '发言内容'], ['STATUS', '審核結果', '状态']]
            _excel_data = _ep.get_row_list(column=_basic_model_columns)
            _idx = 0
            _checking_map = {}
            _has_duplicate = False
            

            _message_parser = MessageParser()

            _excel_data.reverse()
            for _data in _excel_data:
                weight = int(_data[1]) if _data[1] else 1
                msg = _data[2]
                if not msg:
                    continue
                status = int(_data[3])
                is_deleted = status > 0 if status else False
                text, lv, anchor = _message_parser.parse(msg)

                if anchor == 0:
                    _transed = translate_by_string(text)
                    _pinyin_text = ''.join(_transed).replace('_', '')
                    # print('text: ', text)
                    # print('_transed: ', _transed)
                    if len(_pinyin_text) == 1:
                        _pinyin_text = '_'
                    else:
                        _pinyin_text = re.sub(r'\d+', '_', _pinyin_text)
                    
                    _check = _checking_map.get(_pinyin_text, None)
                    if _check:
                        _check_is_deleted = _check[2] > 0
                        if _check_is_deleted != is_deleted:
                            # continue
                            _has_duplicate = True
                            _against_idx = _check[0]
                            _against_msg = _check[1]
                            print('Duplicate MSG [{}], idx: {} || against idx: {}, msg: [{}]'.format(msg, _idx, _against_idx, _against_msg))
                        
                    else:
                        
                        _checking_map[_pinyin_text] = [_idx, msg, is_deleted]

                    _r_idx = self.find_index_of_rlist(msg, result_list)
                    
                    if _r_idx >= 0:
                        if result_list[_r_idx][3] == status:
                            result_list[_r_idx][1] += weight
                        else:
                            print('Confusion Data Index: {} Msg: {}'.format(_r_idx, result_list[_r_idx][2]))
                            print('::  Override By Result: [{}] [{}] Weight: {}'.format(msg, status, weight))
                            # _has_duplicate = True
                            # 後來的資料蓋前面 再增強
                            result_list[_r_idx][3] = status
                            result_list[_r_idx][1] = weight * weight
                    else:
                        weight += 3
                        result_list.append(['', weight, msg, status])

                _idx += 1

            if _has_duplicate:

                print('Stop.')

            else:
                print('_excel_data length: ', len(_excel_data))
                print('result_list length: ', len(result_list))
                print('output_path: ', output_path)

                _jp = JsonParser(file=output_path)
                _jp.save(result_list)
            
        except Exception as err:
            print(err)


        