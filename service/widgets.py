import datetime

def printt(*args):
    _now = datetime.datetime.now()
    _format = '[{}] '.format(_now)
    print(_format, *args, flush=True)