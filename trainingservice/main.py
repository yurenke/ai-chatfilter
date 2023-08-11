from trainingservice.websockets import WebsocketThread
import socketserver, socket
import os, sys, getopt
import logging, time
from service import instance



class LaunchTrainingService():

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
        self.websocket = WebsocketThread("Websocket Thread", host=webhost, port=webport, local_host=self.local_host, on_message_callback=self.on_websocket_message)
        self.websocket_host = webhost
        self.websocket_port = webport

        self.start()

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
            
            if self.websocket.is_active:

                logging.info('TCP Socket connect to Websocket done.')

            self.service_instance.open_mind()
            
        except KeyboardInterrupt:

            self.websocket.stop()

            logging.info('TCP Socket Server Stoped.')

        except Exception as err:
            
            self.websocket.stop()

            logging.error(err)

        sys.exit(2)

