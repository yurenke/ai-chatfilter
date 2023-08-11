from dataparser.apps import ExcelParser
from service.models import Textbook
from dataparser.apps import MessageParser
import re

def insert_textbook(file_path=None):
    # Model = apps.get_model(app_label='service', model_name=model_name)
    # texts = Textbook.objects.values_list('text', flat=True)

    if file_path:

        ep = ExcelParser(file=file_path)
        columns = [['发言内容'], ['STATUS', '審核結果', '状态'], ['類型']]
        result_list = ep.get_row_list(column=columns)
        # print(result_list)

        _i = 0
        _length = len(result_list)

        message_parser = MessageParser()

        for _ in result_list:
            _msg = _[0]
            _status = _[1]
            _mode = _[2]

            if _msg and _status:
                _status = int(_status)
                _mode = _mode.strip()
                _text, lv, anchor = message_parser.parse(_msg)
                if _text:
                    if _status == 0:
                        _model = 0
                    elif _mode == 'g' and len(re.findall(r'[0-9a-zA-Z]', _text)) > 3:
                        _model = 2
                    elif _mode == 'w':
                        continue # 水軍先不處理
                    else:
                        _model =1
                    
                    
                    next_book = Textbook(message=_msg, text=_text, status=_status, model=_model)
                    next_book.save()

            if _i % 50 == 0:
                print('{:2.2f}%'.format(_i / _length * 100), end='\r')
            
            _i += 1

    else:
        pass