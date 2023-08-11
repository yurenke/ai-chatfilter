from __future__ import absolute_import, division, print_function, unicode_literals

import os, json, xlwt
import sys, getopt
import tensorflow as tf
import tensorflow_datasets as tfds

from .helper import print_spend_time, get_pinyin_path
from datetime import datetime
from service.main import MainService
from dataparser.apps import ExcelParser
main_service = MainService(is_admin_server=True)
main_service.open_mind()


def predict_by_ai(text = '', room = '', silence = False, detail=False):
    
    # _text, _lv, _anchor = message_parser.parse(text)

    results = main_service.think(message=text, user='', room=room, silence=silence, detail=detail)
    prediction = results.get('prediction', 0)
    # reason_char = results.get('reason_char', '')
    text = results.get('text', '')
    spend_time = results.get('spend_time', 0)

    if spend_time > 0.35:
        print('Spend Too Much Time of Think. Time({}), Text({}), P({})'.format(spend_time, text, prediction))

    return prediction, text


def predict_by_excel_file(file, silence=True, output_json=False, output_excel=False, status_human_delete=3, status_vendor_ai_delete=5, plus=False):
    status_water_army = 2
    _basic_model_columns = [['VID', '房號'], ['LOGINNAME', '會員號'], ['MESSAGE', '聊天信息', '发言内容'], ['STATUS', '審核結果', '状态'], ['FIX']]
    _status_list = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,21]
    _i = 0

    _st_time = datetime.now()

    ep = ExcelParser(file=file)
    row_list = ep.get_row_list(column=_basic_model_columns)

    num = {
        'total': len(row_list),
        'total_right': 0,
        'total_right_delete': 0,
        'total_wrong': 0,
        'missing_delete': 0,
        'mistake_delete': 0,
        'vendor_ai': 0,
        'human': 0,
        'water_army': 0,
    }

    map = {
        'right': {},
        'wrong': {},
        'mistake_text': {},
    }

    room = 'local'

    next_learning_book = []

    for _s in _status_list:
        map['right'][_s] = 0
        map['wrong'][_s] = 0
        map['mistake_text'][_s] = []

    try:
        
        for row in row_list:
            txt = row[2]
            if not txt:
                continue
            fix = int(row[4]) if str(row[4]).isdigit() else None
            ans = fix if fix is not None and fix >= 0 else int(row[3])
            room = row[0]
            should_be_deleted = ans > 0
            # print(txt)
            predicted, processed_text = predict_by_ai(txt, silence=silence, room=room)
            my_ai_deleted = predicted > 0
            # print(predicted)

            if should_be_deleted == my_ai_deleted or not processed_text:

                num['total_right'] += 1
                map['right'][predicted] += 1
                if should_be_deleted:
                    num['total_right_delete'] += 1
                
            else:

                num['total_wrong'] += 1
                map['wrong'][predicted] += 1

                if should_be_deleted:

                    if ans == status_human_delete or ans == status_water_army:
                        # human delete is right
                        # next_learning_book.append(txt)
                        
                        # human newly record
                        if plus:
                            num['total_right_delete'] += 1
                            num['total_wrong'] -= 1
                            num['total_right'] += 1
                            
                        else:
                            num['missing_delete'] += 1
                            map['mistake_text'][predicted].append(txt)
                    else:
                        num['missing_delete'] += 1
                        map['mistake_text'][predicted].append(txt)

                else:
                    
                    # if processed_text:
                    map['mistake_text'][predicted].append(txt)
                    num['mistake_delete'] += 1

            if ans == status_vendor_ai_delete:
                num['vendor_ai'] += 1
            elif ans == status_human_delete:
                num['human'] += 1
            elif ans == status_water_army:
                num['water_army'] += 1

            _i += 1
            if _i % 100 == 0:
                
                percent = _i / num['total']
                print("Progress of Prediction: {:2.1%}".format(percent), end="\r")

    except KeyboardInterrupt:
        print('KeyboardInterrupt Stop.')

    num['total'] = _i
    
    print('================== Prediction Result ==================')
    print('num_total_legnth: ', _i)
    print('num_total_rights: ', num['total_right'])
    print('num_total_rights_delete: ', num['total_right_delete'])
    print('num_total_wrongs: ', num['total_wrong'])
    print('num_missing_delete: ', num['missing_delete'])
    print('num_mistake_delete: ', num['mistake_delete'])
    print('num_origin_pass: ', _i - (num['vendor_ai'] + num['human']))
    print('num_origin_vendor_ai_delete: ', num['vendor_ai'])
    print('num_origin_human_delete: ', num['human'])
    print('num_water_army: ', num['water_army'])

    ratio_right = "{:2.2%}".format(num['total_right'] /_i)
    ratio_right_delete = "{:2.2%}".format((num['total_right_delete'] / (num['total_right_delete'] + num['missing_delete'])) if num['total_right_delete'] > 0 else 0)
    ratio_mistake_delete = "{:2.2%}".format(num['mistake_delete'] / _i)
    ratio_missing_delete = "{:2.2%}".format(num['missing_delete'] / _i)
    print('ratio right: ', ratio_right)
    print('ratio right delete : ', ratio_right_delete)
    print('ratio missing delete: ', ratio_missing_delete)
    print('ratio mistake delete: ', ratio_mistake_delete)
    print('================== Prediction Details ==================')
    ratio_mistake_map = {}
    print('Mistake Ratios: ')
    for status, wrong_num in map['wrong'].items():
        right_num = map['right'][status]
        _sum = wrong_num + right_num
        if _sum == 0:
            _ratio = 0
        else:
            _ratio = wrong_num / _sum

        _percent = '{:2.2%}'.format(_ratio)
        ratio_mistake_map[status] = _percent
        print(' [{}] = {}   ::  {}/{}'.format(status, _percent, wrong_num, _sum))

    # print(mistake_texts_map)

    if output_json:
        json_data = {
            'num': num,
            'ratio_right': ratio_right,
            'ratio_right_delete': ratio_right_delete,
            'ratio_missing_delete': ratio_missing_delete,
            'ratio_mistake_delete': ratio_mistake_delete,
            'ratio_mistake_map': ratio_mistake_map,
            'map_details': map,
            'learning_book': next_learning_book,
        }
        last_dot = file.rfind('.')
        file_surfix = file[last_dot-4:last_dot] if last_dot > 4 else 'folder'

        json_file_path = os.getcwd() + '/__prediction_{}__.json'.format(file_surfix)

        with open(json_file_path, 'w+', encoding = 'utf8') as handle:
            json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
            handle.write(json_string)

    # 
    if output_excel:
        book = xlwt.Workbook(encoding='utf-8')
        sheet = book.add_sheet("Sheet 1")
        last_dot = file.rfind('.')
        file_surfix = file[last_dot-4:last_dot] if last_dot > 4 else 'folder'
        filename = '__prediction_{}__.xls'.format(file_surfix)
        
        default_width = sheet.col(0).width
        sheet.col(0).width = default_width * 3
        sheet.col(5).width = default_width * 3

        title_style = xlwt.easyxf('pattern: pattern solid, fore_colour gray25;')

        excel_infos = [
            ('ratio_right', ratio_right),
            ('ratio_right_delete', ratio_right_delete),
            ('ratio_mistake_delete', ratio_mistake_delete),
            ('ratio_missing_delete', ratio_missing_delete),
            ('num_rights', num['total_right']),
            ('num_wrongs', num['total_wrong']),
            ('num_total', num['total']),
            ('num_total_rights_delete', num['total_right_delete']),
            ('num_total_missing_delete', num['missing_delete']),
            ('num_total_mistake_delete', num['mistake_delete']),
        ]

        _i = 0
        for ei in excel_infos:
            _label = ei[0]
            _val = ei[1]
            sheet.write(_i,0, _label)
            sheet.write(_i,1, _val)
            _i += 1

        detail_start_row = _i + 3
        should_be_delete_start_column = 0
        sheet.write(detail_start_row,should_be_delete_start_column, '未刪除', style=title_style)
        sheet.write(detail_start_row,should_be_delete_start_column +1, '', style=title_style)
        sheet.write(detail_start_row,should_be_delete_start_column +2, '修正', style=title_style)
        texts_should_be_deleted_but_not = map['mistake_text'][0]
        for i in range(1, len(texts_should_be_deleted_but_not)):
            _row = detail_start_row + i
            _text = texts_should_be_deleted_but_not[i-1]
            sheet.write(_row, should_be_delete_start_column, _text)

        mistake_start_column = 5
        sheet.write(detail_start_row,mistake_start_column, '誤刪', style=title_style)
        sheet.write(detail_start_row,mistake_start_column +1, '狀態', style=title_style)
        sheet.write(detail_start_row,mistake_start_column +2, '修正', style=title_style)
        _row = detail_start_row + 1

        for s in _status_list:
            if s == 0:
                continue
            texts_mistake_deleted = map['mistake_text'][s]
            # sheet.write(_row, mistake_start_column +1, s)
            for _text in texts_mistake_deleted:
                sheet.write(_row, mistake_start_column, _text)
                sheet.write(_row, mistake_start_column +1, s)
                _row += 1

        book.save(filename)

    spend_sec = print_spend_time(_st_time)
    print('Average Time Spent Per Sentence: {:.4f}s'.format(spend_sec / num['total']))
        
    return ratio_right, next_learning_book

