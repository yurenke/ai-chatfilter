from django.core.management.base import BaseCommand
from django.apps import apps
from dataparser.apps import ExcelParser, MessageParser
from dataparser.classes.csv import CsvParser
from dataparser.jsonparser import JsonParser
from ai.classes.translator_pinyin import translate_by_string
import os, json, re
from datetime import date

class Command(BaseCommand):
    help = "parse the excel file to json data."
    _checking_map = {}

    def add_arguments(self, parser):
        parser.add_argument(
            '-i', dest='input_file', required=True,
            help='the path of excel file.',
        )
        parser.add_argument(
            '-o', dest='output_path', required=False,
            help='the name of output.',
        )
        parser.add_argument(
            '-c', dest='check_file_path', required=False,
            help='json file path for check.',
        )
        parser.add_argument(
            '-dup', dest='check_old_data_duplicate', required=False, action='store_true',
        )
        parser.add_argument(
            '-re', dest='revers_data_state', required=False,
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

    def check_duplicate(self, idx, text, is_deleted):
        _transed = translate_by_string(text)
        _pinyin_text = ''.join(_transed).replace('_', '')
        # print('text: ', text)
        # print('_transed: ', _transed)
        if len(_pinyin_text) == 1:
            _pinyin_text = '_'
        else:
            _pinyin_text = re.sub(r'\d+', '_', _pinyin_text)
        
        _check = self._checking_map.get(_pinyin_text, None)
        if _check:
            _check_is_deleted = _check[2] > 0
            if _check_is_deleted != is_deleted:
                _against_idx = _check[0]
                _against_msg = _check[1]
                print('Duplicate text [{}], idx: {} || against idx: {}, msg: [{}]'.format(text, idx, _against_idx, _against_msg))

                return True
        else:
            self._checking_map[_pinyin_text] = [idx, text, is_deleted]
        
        return False

    def handle(self, *args, **options):
        input_file = options.get('input_file')
        output_path = options.get('output_path', None)
        check_file_path = options.get('check_file_path', None)
        check_old_data_duplicate = options.get('check_old_data_duplicate', False)
        revers_data_state = options.get('revers_data_state', None)

        if revers_data_state is None:
            revers_data_state = 0
        else:
            revers_data_state = int(revers_data_state)

        result_list = []
        confusion_list = []

        if output_path is None:
            _dirname = os.path.dirname(input_file)
            _filename = '{}.json'.format(date.today())
            output_path = os.path.join(_dirname, _filename)

        if check_file_path:
            _jp = JsonParser(file=check_file_path)
            _old_json = _jp.load()
            _length_json = len(_old_json)
            _j_idx = -1
            print('')
            for _oj in _old_json:
                _j_idx += 1
                if _j_idx % 100 ==0:
                    print('Handle Old Json.. [ {:.1%} ]'.format(_j_idx / _length_json), end='\r')
                _length_oj = len(_oj)
                _text = _oj[2]

                _oj[2] = _text.strip()
                if len(_text) == 0:
                    continue

                if not check_old_data_duplicate:
                    result_list.append(_oj)
                    continue
                
                weight = int(_oj[1]) if _oj[1] else 1
                msg = _oj[4] if _length_oj>4 else _text
                status = int(_oj[3])

                if self.check_duplicate(_j_idx, msg, status > 0):
                    _r_idx = self.find_index_of_rlist(msg, result_list)
                    if _r_idx >= 0:
                        result_list[_r_idx][1] += weight
                    continue
                
                result_list.append([_oj[0], weight, msg, status])


        # _jp = JsonParser(file=output_path)
        # print('output_path: ', output_path)
        # _jp.save(result_list)
        # exit(2)

        try:
            print('Start Handle Csv.')
            _cp = CsvParser(file=input_file)
            # _basic_model_columns = [['VID', '房號'], ['WEIGHT', '權重'], ['MESSAGE', '聊天信息', '禁言内容', '发言内容'], ['STATUS', '審核結果', '状态']]
            _basic_model_columns = [['MESSAGE', '聊天信息', '禁言内容', '发言内容'], ['STATUS', '審核結果', '状态']]
            _readed_data = _cp.get_row_list(column=_basic_model_columns)
            _parsed_data = [['',1,d[0],int(d[1])] for d in _readed_data]
            # print('_parsed_data: ', _parsed_data[:5])
            _idx = 0
            _has_duplicate = False

            _message_parser = MessageParser()

            # _parsed_data.reverse()
            for _data in _parsed_data:
                weight = int(_data[1]) if _data[1] else 1
                msg = _data[2]
                status = int(_data[3])
                is_deleted = status > 0 if status else False
                text, lv, anchor = _message_parser.parse(msg)
                text = text.strip()
                if text and anchor == 0 and lv < 6:
                    
                    ###

                    _r_idx = self.find_index_of_rlist(text, result_list)
                    
                    if _r_idx >= 0:
                        _old_status = result_list[_r_idx][3]
                        if (_old_status > 0 and is_deleted) or (_old_status == 0 and not is_deleted):
                            result_list[_r_idx][1] += weight
                        else:
                            print('Confusion Data Index: {} Msg: {} State: {} New State: {}'.format(_r_idx, result_list[_r_idx][2], _old_status, status))
                            # _has_duplicate = True
                            result_list[_r_idx][3] = status
                            result_list[_r_idx][1] = -1
                            confusion_list.append([text, '{}=>{}'.format(_old_status, status)])
                    else:
                        result_list.append([_data[0], weight, text, status])

                _idx += 1

            _jp = JsonParser(file=output_path)
            print('output_path: ', output_path)

            if _has_duplicate:

                print('Stop.')
                print('confusion_list length: ', len(confusion_list))
                _jp.save(confusion_list)

            else:
                print('_parsed_data length: ', len(_parsed_data))
                result_list = [_ for _ in result_list if _[1]>0]
                print('result_list length: ', len(result_list))
                _jp.save(result_list)
            
            
        except Exception as err:
            print(err)


        