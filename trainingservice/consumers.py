from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import async_to_sync
from service import instance
from django.apps import apps

# from django.conf import settings
import json
from service.widgets import printt




class ChatConsumer(AsyncWebsocketConsumer):
    """

    """

    main_service = None

    hostname = None
    has_admin_client = False
    is_tcp = False
    is_standby = True
    is_working = False
    group_name_global = 'GLOBAL_CHATTING'
    group_name_admin_client = 'GLOBAL_CHATTING_ADMIN_CLIENT'

    key_get_model = '__getmodel__'
    key_is_admin_client = '__isadminclient__'
    key_tcp_poto = '__tcp__'
    key_change_nickname_request = '__changenicknamerequest__'
    key_training_start = '__trainingstart__'
    key_tcpsocket_connection_login = '__tcpsocketconnectionlogin__'

    def check_service(self):
        if not self.main_service:
            self.main_service = instance.get_main_service(is_admin=True)
        
        if not self.main_service.is_open_mind:
            self.main_service.open_mind()
    

    async def connect(self):
        # self.room_name = self.scope['url_route']['kwargs']['room_name']
        # async_to_sync(self.channel_layer.group_add)(
        #     self.group_name_global,
        #     self.channel_name
        # )
        self.check_service()

        await self.channel_layer.group_add(
            self.group_name_global,
            self.channel_name
        )

        print('== Consumer connected == channel_name: ', self.channel_name, flush=True)

        await self.accept()


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name_global,
            self.channel_name
        )

        await self.channel_layer.group_discard(
            self.group_name_admin_client,
            self.channel_name
        )


    async def receive(self, text_data):
        text_data_json = json.loads(text_data)

        msgid = text_data_json.get('msgid', None)
        message = text_data_json.get('message', None)
        

        if message and isinstance(msgid, int):

            user = text_data_json.get('user', None)
            room = text_data_json.get('room', None)
            detail = text_data_json.get('detail', False)
            prediction = text_data_json.get('prediction', None)

            result_next = {
                'type': 'channel_chat_message',
                'msgid': msgid,
                # 'message': message,
            }

            if detail:
                
                results = self.main_service.think(message=message, room=room, detail=detail)
                result_next.update(results)
                # printt('websocket ai think: ', results)
            elif isinstance(prediction, int):

                result_next['prediction'] = prediction
                result_next['message'] = message
                result_next['room'] = room

            else:
                
                result_next['prediction'] = 0
                printt('Web Socket [receive] Wrong!! Not Pass Prediction. msgid: ', msgid)
            
            await self.channel_layer.group_send(
                self.group_name_admin_client,
                result_next,
            )
            
            self.main_service.saveRecord(result_next['prediction'], message=message)
            printt('Save Message: {} | {}'.format(message, result_next['prediction']))
        
        elif isinstance(msgid, str):

            message = self.get_message_by_order(msgid, text_data_json)
            await self.do_something_by_order(msgid, message)

        else:
            printt('Not Message msgid: ', msgid)
            
        
        printt('Consumer Receive: ', text_data_json)
        
        

    async def channel_chat_message(self, event):
        
        msgid = event['msgid']

        message = event.get('message', '')
        printt('channel_chat_message: ' + message)
        prediction = int(event.get('prediction', 0))
        user = event.get('user', '')
        room = event.get('room', 'none')
        reason_char = event.get('reason_char', '')
        detail = event.get('detail', {})
        await self.send(text_data=json.dumps({
            'msgid': msgid,
            'message': message,
            'prediction': prediction,
            'reason_char': reason_char,
            'user': user,
            'room': room,
            'detail': detail,
        }))

    
    async def channel_chat_nickname(self, event):
        _nickname = event.get('nickname', False)
        if _nickname:
            await self.send(text_data=json.dumps({
                'msgid': self.key_change_nickname_request,
                'nickname': _nickname,
                'code': event.get('code', 0),
            }))
    

    def get_message_by_order(self, order_key, json = {}):
        if order_key == self.key_tcp_poto:
            return json.get('hostname', None)
        # elif order_key == self.key_send_train_remotely:
        #     return self.main_service.get_train_textbook()
        elif order_key == self.key_get_model:
            _model_name = json.get('model', None)
            _app_name = json.get('app', 'service')
            if _model_name:
                Model = apps.get_model(app_label=_app_name, model_name=_model_name)
                if Model:
                    return list(Model.objects.all()[:1000])
        
        return json


    async def do_something_by_order(self, order_key, message):
        if order_key == self.key_is_admin_client:

            await self.channel_layer.group_add(
                self.group_name_admin_client,
                self.channel_name
            )
            self.is_working = True
            self.has_admin_client = True

        elif order_key == self.key_tcp_poto:

            _hostname = message
            if _hostname:
                self.hostname = _hostname

                self.is_tcp = True

        # elif order_key == self.key_send_train_remotely:
        elif order_key == self.key_change_nickname_request:
            _nickname = message.get('message', '')
            _code = message.get('prediction', None)

            self.main_service.saveNicknameRequestRecord(nickname=_nickname, status=_code)

            await self.channel_layer.group_send(
                self.group_name_admin_client,
                {
                    'type': 'channel_chat_nickname',
                    'nickname': _nickname,
                    'code': _code,
                },
            )
            return None

        elif order_key == self.key_training_start:
            pass
            
        
        await self.send(text_data=json.dumps({'msgid': order_key, 'message': message}))

