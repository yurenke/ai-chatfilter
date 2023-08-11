from tcpsocket.to_websocket import WebsocketThread
from tcpsocket.tcp import socketTcp


import socketserver, socket
import os, sys, getopt
import logging, time
# sys.path.append("..")
from service import instance



class LaunchTcpSocket():

    STATIC_MSGID_REFRESH_PINYIN_BLOCK = '__tcpsocketrefreshpinyinblock__'
    STATIC_MSGID_REFRESH_ALERT_WORDS = '__tcpsocketrefreshalertwords__'
    STATIC_MSGID_REFRESH_NICKNAME_PINYIN_BLOCK = '__tcpsocketrefreshnicknamepinyinblock__'

    addr = (None, None)
    server = None
    service_instance = None
    nickname_filter_instance = None
    websocket = None
    websocket_host = None
    websocket_port = None
    ws_url = ''
    remote_vocabulary = []
    local_host = (None, None)
    is_tcp_connecting = False


    def __init__(self, host, port, webhost, webport):
        # print('[LaunchTcpSocket] hots: {} | port: {}'.format(host, port))
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        self.local_host = (host_name, host_ip)

        self.addr = (host, port)
        self.websocket = WebsocketThread("Websocket Thread-1", host=webhost, port=webport, local_host=self.local_host, on_message_callback=self.on_websocket_message)
        self.websocket_host = webhost
        self.websocket_port = webport
        self.ws_url = self.websocket.get_ws_url()

        self.start()

    
    def handler_factory(self):
        callback = self.handle_tcp_callback
        on_open = self.handle_tcp_open
        on_close = self.handle_tcp_close
        service_instance = self.service_instance
        nickname_filter_instance = self.nickname_filter_instance
        
        def createHandler(*args, **keys):
            return socketTcp(callback, service_instance, on_open, on_close, nickname_filter_instance, *args, **keys)
        return createHandler
    

    def handle_tcp_callback(self, data, prediction, status_code = 0, is_suspicious = 0):
        if prediction is None:
            return
        
        if self.websocket and self.websocket.is_active:

            if data.cmd == 0x040003 or data.cmd == 0x041003:
                #
                _msgid = data.msgid
                _msg = data.msg
                _room = data.roomid if hasattr(data, 'roomid') else ''
                _user = data.loginname if hasattr(data, 'loginname') else ''
                
                self.websocket.send_msg(msgid=_msgid, msg=_msg, room=_room, user=_user, prediction=prediction, is_suspicious=is_suspicious)
                
            elif data.cmd == 0x040007:
                # nickname change
                _nickname = data.nickname
                logging.debug('[handle_tcp_callback][Send To Websocket Nickname] prediction: {} | type: {}'.format(prediction, type(prediction)))
                logging.debug('_nickname: [{}] | type: {}'.format(_nickname, type(_nickname)))
                self.websocket.send_msg(msgid=self.websocket.key_change_nickname_request, msg=_nickname, prediction=prediction)
                
        else:
            if data.cmd == 0x040003 or data.cmd == 0x041003:
                _msg = data.msg
            elif data.cmd == 0x040007:
                _msg = data.nickname
            else:
                _msg = 'None'

            self.service_instance.saveRecord(prediction, _msg)
            
            logging.error('Websocket is Not Working. [txt: {}] [{}]'.format(_msg, prediction))

    
    def handle_tcp_open(self):
        self.is_tcp_connecting = True


    def handle_tcp_close(self):
        self.is_tcp_connecting = False


    def on_websocket_message(self, msgid, message):
        logging.debug('on_websocket_message msgid: {}  message: {}'.format(msgid, message))
        if msgid == self.STATIC_MSGID_REFRESH_PINYIN_BLOCK:
            self.service_instance.reload_pinyin_block()
        elif msgid == self.STATIC_MSGID_REFRESH_ALERT_WORDS:
            self.service_instance.reload_alert_words()
        elif msgid == self.STATIC_MSGID_REFRESH_NICKNAME_PINYIN_BLOCK:
            self.nickname_filter_instance.reload_nickname_pinyin_block()
        # self.service_instance
    

    def start(self):
        try:

            self.websocket.start()
            _max_times = 50
            logging.info('Watting For Connecting.. (Tcpsocket to Websocket).')
            while not self.websocket.is_active:
                time.sleep(0.5)
                _max_times -= 1
                if _max_times <= 0:
                    logging.error('Connection of (Tcpsocket to Websocket) Failed.')
                    break
            
            
            self.service_instance = instance.get_main_service(is_admin=False)
            self.nickname_filter_instance = instance.get_nickname_filter(is_admin=False)
            # self.service_instance = instance.get_main_service(is_admin=True)
            
            if self.websocket.is_active:

                logging.info('TCP Socket connect to Websocket done.')
                self.service_instance.fetch_ai_model_data(remote_ip=self.websocket_host, port=self.websocket_port)
                self.nickname_filter_instance.fetch_nickname_ai_model_data(remote_ip=self.websocket_host, port=self.websocket_port)
            
            
            self.service_instance.open_mind()
            self.nickname_filter_instance.open_mind()
            
            
            # self.nickname_filter_instance.set_english_parser(self.service_instance.get_english_parser())

            self.server = socketserver.ThreadingTCPServer(self.addr, self.handler_factory(), bind_and_activate=True)
            # self.server = socketserver.TCPServer(self.addr, self.handler_factory(), bind_and_activate=True)
            # self.server.request_queue_size = 8
            logging.info('TCP Socket Server launched on port :: {}'.format(self.addr[1]))
            self.server.serve_forever()

        except KeyboardInterrupt:

            self.websocket.stop()

            logging.info('TCP Socket Server Stoped.')

        except Exception as err:
            
            self.websocket.stop()

            logging.error(err)

        if self.server:
            self.server.server_close()
            print('Shutdown Server.')
            self.server.shutdown()
        sys.exit(2)



host = '0.0.0.0'
port = 8025
web_socket_host = '127.0.0.1'
web_socket_port = 8000

# main_service = instance.get_main_service()


if __name__ == '__main__':

    argvs = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argvs, "hp:w:")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o == "-p":
            port = int(a)
        if o == '-w':
            web_socket_port = int(a)

    
    main = LaunchTcpSocket(host, port, web_socket_host, web_socket_port)

    # server = socketserver.TCPServer(addr, socketTcp)
    

    
    
    
    

    
    