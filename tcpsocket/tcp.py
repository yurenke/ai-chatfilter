
from configparser import RawConfigParser
from dataparser.apps import ExcelParser
from socketserver import StreamRequestHandler as Tcp
from tcpsocket.chat_package import pack, unpack
# from queue import Queue
import logging
import os, time, json, threading
# from multiprocessing import Process

SOCKET_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SOCKET_DIR)
MAX_SOCKET_SIZE = 2048

config_setting = RawConfigParser()
config_setting.read(BASE_DIR+'/setting.ini')
config_keys = RawConfigParser()
config_keys.read(SOCKET_DIR+'/keys.cfg')
config_version = RawConfigParser()
config_version.read(BASE_DIR+'/version.cfg')

version = '{}.{}'.format(config_version.get('MAIN', 'V'), config_version.get('MAIN', 'SUR'))
serverid = config_keys.get('SERVER', 'ID')
sig = config_keys.get('SERVER', 'PWD')

logging_level = logging.DEBUG if bool('True' in config_setting.get('MAIN', 'DEBUG')) else logging.INFO
logging.basicConfig(format='[%(levelname)s]%(asctime)s %(message)s', datefmt='(%m/%d) %I:%M:%S %p :: ', level=logging_level)

threadLock = threading.Lock()
# threadLimiter = threading.BoundedSemaphore(8)

SECOND_TIMEOUT_THREADING = 1

class LockThread(threading.Thread):
    def __init__(self, target, args):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        try:
            if threadLock.acquire(timeout=1):
            # if threadLimiter.acquire(timeout=1):
                self.target(*self.args)
            else:
                logging.error('Timeout Threading With Thread Count [ {} ]'.format(threading.active_count()))
                self.target(*self.args, True)

            threadLock.release()
        except Exception as exp:

            logging.error('LockThread Failed. [ {} ]'.format(exp))

        finally:
            
            del self.target, self.args
            

# class testPureTcp(Tcp):
#     length_done = 0
#     last_time = time.time()
#     def __init__(self, *args, **keys):
#         super().__init__(*args, **keys)
#     def handle(self):
#         print('::TestPureTcp Start handler_factory')
#         self.last_time = time.time()
#         while True:
            
#             # readed = self.rfile.read(8096)
#             # print('readed: ', readed.decode("utf-8", errors='ignore') )
#             recived = self.request.recv(512)
#             _, _left = unpack(recived)
#             now = time.time()
#             print('recived time gap: {}'.format(now - self.last_time))
#             print('recived: ', recived.decode("utf-8", errors='ignore'))
#             print('size: ', _.size)
#             print('=============================')
#             self.last_time = now
#             self.length_done += 1
#             if not recived:
#                 logging.error('Not Recived [ {} ]'.format(recived))
#                 break
            
#             try:
#                 packed_res = pack(0x040004, msgid=1, code=5)
#                 self.request.send(packed_res)
#             except Exception as exp:
#                 logging.error('Request Sendall Failed. exp[ {} ]'.format(exp))
        
#         logging.info('::TestPureTcp End handler_factory Done Recived: {}'.format(self.length_done))


class socketTcp(Tcp):

    callback = None
    on_client_open = None
    on_client_close = None
    service_instance = None
    nickname_filter_instance = None
    left_byte = b''
    # thread_queue = Queue(4)
    

    def __init__(self, callback, service_instance, on_client_open = None, on_client_close = None, nickname_filter_instance = None, *args, **keys):
        self.callback = callback
        self.on_client_open = on_client_open
        self.on_client_close = on_client_close
        self.service_instance = service_instance
        self.nickname_filter_instance = nickname_filter_instance
        
        super().__init__(*args, **keys)


    def handle_recive_threading(self, recived, directly_reject = False):
        unpacked_data, unpacked_left_buffer = unpack(recived)

        status_code = -1
        prediction = None
        is_suspicious = 0
        if unpacked_data.size == 0:
            _length_unpacked_left_buffer = len(unpacked_left_buffer)
            logging.debug('Handle Recived Stream Byte Again Threading Count: ( {} )  Left Buffer Size: ( {} )'.format(threading.active_count(), _length_unpacked_left_buffer))
            if _length_unpacked_left_buffer > 256:
                self.left_byte = b''
                prediction = 98
            else:
                self.left_byte = unpacked_left_buffer
                return False

        if unpacked_data.cmd == 0x000001:
            logging.debug('Recived [ Check Hearting ]')
            packed_res = pack(0x000001) # hearting

        elif unpacked_data.cmd == 0x041003:
            logging.debug('Recived [ChatJson]  id: {} json: {}'.format(unpacked_data.msgid, unpacked_data.jsonstr))
            status_code = 0
            _msg = unpacked_data.msg

            if isinstance(unpacked_data.msgid, int) and _msg:
                if directly_reject:
                    prediction = 99
                elif prediction is None:
                    ai_results = self.service_instance.think(message=_msg, user=unpacked_data.loginname, room=unpacked_data.roomid)
                    prediction = ai_results.get('prediction', None)
                    is_suspicious = ai_results.get('is_suspicious', 0)
            else:
                logging.error('Parse Failed msgid = {} Txt = {}'.format(unpacked_data.msgid, unpacked_data.msg))
            
            if prediction and prediction > 0:
                status_code = 5
                logging.info('Message be Blocked id: {} msg: {}'.format(unpacked_data.msgid, _msg))
            
            packed_res = pack(0x040004, msgid=unpacked_data.msgid, code=status_code)

        elif unpacked_data.cmd == 0x040001:
            logging.debug('Recived [ Login ]')
            
            is_matched = serverid == unpacked_data.serverid and sig == unpacked_data.sig

            if is_matched:
                logging.debug('Login Successful serverid: {}'.format(serverid))
                server_code = 0
            else:
                logging.debug('Login Failed. unpacked.sig: {}'.format(unpacked_data.sig))
                server_code = 1

            packed_res = pack(0x040002, code=server_code)

        elif unpacked_data.cmd == 0x040002:
            logging.debug('Recived [ Login Response ]')
            packed_res = pack(0x000001)

        elif unpacked_data.cmd == 0x040003:
            logging.debug('Recived [Chat] msg: {}'.format(unpacked_data.msg))

            status_code = 0
            _msg = unpacked_data.msg
            if isinstance(unpacked_data.msgid, int) and _msg:
                ai_results = self.service_instance.think(message=_msg)
                prediction = ai_results.get('prediction', None)
                is_suspicious = ai_results.get('is_suspicious', 0)
            
            if prediction:
                if prediction > 0:
                    status_code = 5
                    logging.info('Message be blocked = id: {} txt: {}'.format(unpacked_data.msgid, unpacked_data.msg))

            packed_res = pack(0x040004, msgid=unpacked_data.msgid, code=status_code)

        elif unpacked_data.cmd == 0x040004:
            logging.debug('Recived [ Chat Response ]')
            # logging.debug('msgid: {}'.format(unpacked_data.msgid))
            # logging.debug('code: {}'.format(unpacked_data.code))
            packed_res = pack(0x000001)

        elif unpacked_data.cmd == 0x040007:
            logging.debug('Recived [ Nickname Change Request ] id: {} | name: {} | size: {}'.format(unpacked_data.reqid, unpacked_data.nickname, unpacked_data.size))

            if self.nickname_filter_instance:

                ai_results = self.nickname_filter_instance.think(nickname=unpacked_data.nickname)
                code = ai_results.get('code', 0)
                is_suspicious = 0

                if code and code > 0:
                    logging.info('Nickname Change Request be blocked = reqid: {} | code: {}'.format(unpacked_data.reqid, code))

            else:

                code = 0

            packed_res = pack(0x040008, reqid=unpacked_data.reqid, code=code)
            prediction = code

        else:
            
            logging.error('Recived Package Unknow.  [ {} ]'.format(recived.decode("utf-8", errors='ignore')))
            return False
        
        try:
            # _, _left = unpack(packed_res)
            # if _.cmd == 0x040004:
            #     logging.debug('Test pack before send byte: [ {} ]'.format(packed_res))
            #     logging.debug('Before Unpack cmd: [ {} ] msgid: [ {} ] code: [ {} ]'.format(_.cmd, _.msgid, _.code))
            self.request.sendall(packed_res)
            if self.callback and prediction is not None:
                self.callback(unpacked_data, int(prediction), status_code, is_suspicious)
        except Exception as exp:
            logging.error('Request Sendall Failed.')
            print(exp)
        
        if unpacked_left_buffer:
            return unpacked_left_buffer
        else:
            return False
    


    def handle(self):
        logging.info('**TCPSocket Version[{}] clinet has connected, address: {}'.format(version, self.client_address))
        self.on_client_open()
        _is_holding_size = True
        _half_size = MAX_SOCKET_SIZE / 2
        while True:
            
            recived = self.request.recv(MAX_SOCKET_SIZE)
            if not recived:
                logging.error('Not Recived [ {} ]'.format(recived.decode("utf-8", errors='ignore')))
                break
            _start_handle_recived_time = time.time()
            # _thread = LockThread(target = self.handle_recive_threading, args = (recived,))
            # _thread.start()
            if self.left_byte:
                recived = self.left_byte + recived
                self.left_byte = b''

            _is_holding_size = len(recived) < _half_size
            _not_holding = not _is_holding_size
            
            while recived:
                _inner_recived_time = time.time()
                _is_over_time = (_inner_recived_time - _start_handle_recived_time) > SECOND_TIMEOUT_THREADING
                recived = self.handle_recive_threading(recived, _not_holding or _is_over_time)
        
        logging.info('**TCPSocket clinet disconnected, address: {}'.format(self.client_address))
        self.on_client_close()
    

    def handle_error(self):
        logging.error('TCPSocket Handle Error!!')
        self.on_client_close()

    def server_close(self):
        logging.error('TCPSocket Server Closed!!!')
        self.on_client_close()

