import time, sys, getopt, json
import websocket
import threading
# from multiprocessing.connection import Listener
from multiprocessing.pool import ThreadPool
# import asyncio
import logging
# logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d | %I:%M:%S %p :: ', level=logging.DEBUG)

WS_URL =  "ws://{}:{}/ws/chat/"
# websocket.enableTrace(True)

class WebsocketThread (threading.Thread):

    _name = ''
    _port = 80
    _url = ''
    _waitting_ids = []
    # _message_result = dict()
    _limted_timeout = 5

    stop_event = None
    ws = None
    pool = None
    is_active = False
    # second_warn_spend_time = 0.35
    key_tcp_poto = '__tcp__'
    key_change_nickname_request = '__changenicknamerequest__'
    # key_send_train_remotely = '__remotetrain__'
    cache_map = {}

    on_message_callback = None

    local_host = (None, None)


    def __init__(self, name = 'default', host = '0.0.0.0', port = 80, local_host=(None, None), on_message_callback = None):
        threading.Thread.__init__(self)
        self.local_host = local_host
        self._name = name
        self._port = port
        self._url = WS_URL.format(host, port)
        self.stop_event = threading.Event()
        self.pool = ThreadPool(processes=4)
        self.on_message_callback = on_message_callback
    

    def run(self):
        self.on_start()


    def on_start(self):
        logging.info('[WebsocketThread] On Start Websocket URL: {}'.format(self._url))
        self.ws = websocket.WebSocketApp(self._url,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        self.ws.on_open = self.on_open
        
        
        while True:
            if self.stopped():
                break
            self.ws.run_forever(ping_interval=10, ping_timeout=2)
            time.sleep(3)

    
    def on_open(self):
        logging.info('[WebsocketThread] Web Socket Connection opened.')
        self.setting()
        self.is_active = True


    def on_message(self, message):
        # print('on_message', message)
        _json = json.loads(message)
        _msg_id = _json.get('msgid', None)
        
        if _msg_id:
            _res_message = _json.get('message', {})

            if _msg_id in self._waitting_ids:
                self._waitting_ids.remove(_msg_id)

            if self.on_message_callback:
                self.on_message_callback(_msg_id, _res_message)
        

    def on_error(self, error):
        logging.error('### Web Socket Error: {}'.format(error))
        self.is_active = False
        self._waitting_ids = []
        # self._message_result = dict()
        # raise Exception(error)


    def on_close(self):
        self.is_active = False
        logging.warning("# Web Socket Closed ###")
        

    def stop(self):
        self.is_active = False
        self.stop_event.set()
        self.ws.close()


    def stopped(self):
        return self.stop_event.is_set()


    def setting(self):
        return self.pool.apply(self.send_thread, [{'tcp': True, 'msgid': self.key_tcp_poto, 'hostname': self.local_host[0]+self.local_host[1]}])


    def get_ws_url(self):
        return self._url


    def send_thread(self, data):
        _msg_id = data.get('msgid', None)
        _timeout = self._limted_timeout
        _now = time.time()
        _res = {}
        #
        if _msg_id:
            # print('send_thread: ', data)
            _str = json.dumps(data)
            self.ws.send(_str)

            
            if isinstance(_msg_id, int):
                self._waitting_ids.append(_msg_id)
            elif _msg_id == self.key_tcp_poto:
                logging.info('TCP Socket Setting Done.')
                pass
            elif _msg_id == self.key_change_nickname_request:
                pass
            else:
                # msgid is a string order
                self.cache_map[_msg_id] = None
            
                _gap = 0

                while True:
                    _r = self.cache_map.get(_msg_id)
                    _gap = time.time() - _now

                    if _r:
                        _res = _r
                        break
                    
                    if _gap > _timeout:
                        logging.error('### Web Socket Timeout.. msgid:[ {} ]'.format(_msg_id))
                        _res = {
                            'msgid': _msg_id,
                            'message': [],
                        }
                        break

                    time.sleep(0.02)

        else:

            logging.info('Web Socket Do Not Sending. [without msg id]')

        return _res

    
    def send_msg(self, msgid, msg, room='', user='', prediction=0):
        _data = {
            'message':msg,
            'msgid':msgid,
            'room':room,
            'user':user,
            'prediction': prediction,
        }
        # print('thinking start run!!')

        values = self.pool.apply(self.send_thread, [_data])

        # print('thinking end!! ... values: ', values)
        
        return values



    
    

