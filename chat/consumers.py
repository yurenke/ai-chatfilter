from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import AcceptConnection, DenyConnection, InvalidChannelLayerError, StopConsumer
from asgiref.sync import async_to_sync, sync_to_async
from service import instance
from django.apps import apps

# from django.conf import settings
import json
from service.widgets import printt




class ChatConsumer(AsyncWebsocketConsumer):
    """

    """

    main_service = None
    nickname_filter = None

    hostname = None
    has_admin_client = False
    is_tcp = False
    is_standby = True
    is_working = False
    group_name_global = 'GLOBAL_CHATTING'
    group_name_admin_client = 'GLOBAL_CHATTING_ADMIN_CLIENT'
    group_name_tcpsocket_client = 'GLOBAL_CHATTING_TCPSOCKET_CLIENT'

    key_get_model = '__getmodel__'
    key_is_admin_client = '__isadminclient__'
    key_tcp_poto = '__tcp__'
    key_change_nickname_request = '__changenicknamerequest__'
    key_training_start = '__trainingstart__'
    key_tcpsocket_connection_login = '__tcpsocketconnectionlogin__'
    key_tcp_refresh_pinyin_block = '__tcpsocketrefreshpinyinblock__'
    key_tcp_refresh_nickname_pinyin_block = '__tcpsocketrefreshnicknamepinyinblock__'
    key_tcp_refresh_alert_words = '__tcpsocketrefreshalertwords__'

    STATIC_REFRESH = '_REFRESH_'

    def check_service(self):
        if not self.main_service:
            self.main_service = instance.get_main_service(is_admin=True)
        
        if not self.nickname_filter:
            self.nickname_filter = instance.get_nickname_filter(is_admin=True)
        
        if not self.main_service.is_open_mind:
            self.main_service.open_mind()

        if not self.nickname_filter.is_open_mind:
            self.nickname_filter.open_mind()
    
    async def websocket_connect(self, message):
        """
        Called when a WebSocket connection is opened.
        """
        await sync_to_async(self.check_service)()
        #self.groups.append(self.group_name_global)

        await self.channel_layer.group_add(
            self.group_name_global,
            self.channel_name
        )

        print('== Consumer connected == channel_name: ', self.channel_name, flush=True)

        try:
            await self.connect()
        except AcceptConnection:
            await self.accept()           
        except DenyConnection:
            await self.close()

    # async def connect(self):
    #     # self.room_name = self.scope['url_route']['kwargs']['room_name']
    #     # async_to_sync(self.channel_layer.group_add)(
    #     #     self.group_name_global,
    #     #     self.channel_name
    #     # )
    #     self.check_service()

    #     await self.channel_layer.group_add(
    #         self.group_name_global,
    #         self.channel_name
    #     )

    #     print('== Consumer connected == channel_name: ', self.channel_name, flush=True)

    #     await self.accept()

    async def websocket_disconnect(self, message):
        """
        Called when a WebSocket connection is closed. Base level so you don't
        need to call super() all the time.
        """
        try:
            await self.channel_layer.group_discard(
                self.group_name_global,
                self.channel_name
            )

            await self.channel_layer.group_discard(
                self.group_name_admin_client,
                self.channel_name
            )

            await self.channel_layer.group_discard(
                self.group_name_tcpsocket_client,
                self.channel_name
            )
        except AttributeError:
            raise InvalidChannelLayerError(
                "BACKEND is unconfigured or doesn't support groups"
            )
        await self.disconnect(message["code"])
        raise StopConsumer()


    # async def disconnect(self, close_code):
    #     await self.channel_layer.group_discard(
    #         self.group_name_global,
    #         self.channel_name
    #     )

    #     await self.channel_layer.group_discard(
    #         self.group_name_admin_client,
    #         self.channel_name
    #     )

    #     await self.channel_layer.group_discard(
    #         self.group_name_tcpsocket_client,
    #         self.channel_name
    #     )


    async def receive(self, text_data):
        text_data_json = json.loads(text_data)

        msgid = text_data_json.get('msgid', None)
        message = text_data_json.get('message', None)
        

        if message and isinstance(msgid, int):

            user = text_data_json.get('user', None)
            room = text_data_json.get('room', None)
            detail = text_data_json.get('detail', False)
            prediction = text_data_json.get('prediction', None)
            is_suspicious = text_data_json.get('is_suspicious', 0)

            result_next = {
                'type': 'channel_chat_message',
                'msgid': msgid,
                'user': user,
                # 'message': message,
            }

            if detail:
                
                results = self.main_service.think(message=message, room=room, user=user, detail=detail)
                _text, _lv, _anchor = self.main_service.parse_message(message)
                result_next.update(results)
                result_next['text'] = _text
                result_next['lv'] = _lv
                result_next['anchor'] = _anchor
                # printt('check consumer receive result_next: ', result_next)
                # printt('websocket ai think: ', results)
                # printt('websocket result_next: ', result_next)
            elif isinstance(prediction, int):
                _text, _lv, _anchor = self.main_service.parse_message(message)
                result_next['prediction'] = prediction
                # result_next['message'] = message
                result_next['text'] = _text
                result_next['lv'] = _lv
                result_next['anchor'] = _anchor
                result_next['room'] = room
                result_next['user'] = user
                result_next['is_suspicious'] = is_suspicious

            else:
                
                result_next['prediction'] = 0
                printt('Web Socket [receive] Wrong!! Not Pass Prediction. msgid: ', msgid)
            
            await self.channel_layer.group_send(
                self.group_name_admin_client,
                result_next,
            )
            
            await sync_to_async(self.main_service.saveRecord)(result_next['prediction'], message=message)
            printt('Save Message: {} | {}'.format(message, result_next['prediction']))
        
        elif isinstance(msgid, str):

            # message = await sync_to_async(self.get_message_by_order)(msgid, text_data_json)
            # await self.do_something_by_order(msgid, message)
            await self.do_something_by_order(msgid, text_data_json)

        else:
            printt('Not Message msgid: ', msgid)
            
        
        printt('Consumer Receive: ', text_data_json)
        
        

    async def channel_chat_message(self, event):
        
        msgid = event['msgid']

        message = event.get('message', '')
        text = event.get('text', '')
        prediction = int(event.get('prediction', 0))
        user = event.get('user', '')
        room = event.get('room', 'none')
        reason_char = event.get('reason_char', '')
        detail = event.get('detail', {})
        await self.send(text_data=json.dumps({
            'msgid': msgid,
            'text': text,
            'message': message,
            'prediction': prediction,
            'reason_char': reason_char,
            'user': user,
            'room': room,
            'detail': detail,
            'lv': int(event.get('lv', 0)),
            'anchor': int(event.get('anchor', 0)),
            'is_suspicious': int(event.get('is_suspicious', 0))
        }))

    
    async def channel_chat_nickname(self, event):
        _nickname = event.get('nickname', False)
        reason_char = event.get('reason_char', '')
        detail = event.get('detail', {})
        if _nickname:
            await self.send(text_data=json.dumps({
                'msgid': self.key_change_nickname_request,
                'nickname': _nickname,
                'code': event.get('code', 0),
                'reason_char': reason_char,
                'detail': detail
            }))

    async def channel_chat_to_tcpsocket(self, event):
        await self.send(text_data=json.dumps({
            'msgid': event.get('msgid', ''),
            'message': event.get('message', 0),
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


    # async def do_something_by_order(self, order_key, message):
    async def do_something_by_order(self, order_key, text_json = {}):
        if order_key == self.key_tcp_poto:
            message = text_json.get('hostname', None)
        # elif order_key == self.key_send_train_remotely:
        #     return self.main_service.get_train_textbook()
        elif order_key == self.key_get_model:
            _model_name = text_json.get('model', None)
            _app_name = text_json.get('app', 'service')
            if _model_name:
                Model = apps.get_model(app_label=_app_name, model_name=_model_name)
                if Model:
                    message = list(Model.objects.all()[:1000])
        
        else:
            message = text_json    

        if order_key == self.key_is_admin_client:
            # self.groups.append(self.group_name_admin_client)

            await self.channel_layer.group_add(
                self.group_name_admin_client,
                self.channel_name
            )
            self.is_working = True
            self.has_admin_client = True

        elif order_key == self.key_tcp_poto:

            _hostname = message
            # self.groups.append(self.group_name_tcpsocket_client)
            # print('_hostname: ', _hostname)
            if _hostname:
                await self.channel_layer.group_add(
                    self.group_name_tcpsocket_client,
                    self.channel_name
                )
                self.hostname = _hostname

                self.is_tcp = True

        elif order_key == self.key_tcp_refresh_pinyin_block:
            await self.channel_layer.group_send(
                self.group_name_tcpsocket_client,
                {
                    'type': 'channel_chat_to_tcpsocket',
                    'msgid': self.key_tcp_refresh_pinyin_block,
                    'message': 1,
                },
            )
            await sync_to_async(self.main_service.reload_pinyin_block)()
            return None

        elif order_key == self.key_tcp_refresh_nickname_pinyin_block:
            await self.channel_layer.group_send(
                self.group_name_tcpsocket_client,
                {
                    'type': 'channel_chat_to_tcpsocket',
                    'msgid': self.key_tcp_refresh_nickname_pinyin_block,
                    'message': 1,
                },
            )
            await sync_to_async(self.nickname_filter.reload_nickname_pinyin_block)()
            return None

        elif order_key == self.key_tcp_refresh_alert_words:
            await self.channel_layer.group_send(
                self.group_name_tcpsocket_client,
                {
                    'type': 'channel_chat_to_tcpsocket',
                    'msgid': self.key_tcp_refresh_alert_words,
                    'message': 1,
                },
            )
            await sync_to_async(self.main_service.reload_alert_words)()
            return None

        # elif order_key == self.key_send_train_remotely:
        elif order_key == self.key_change_nickname_request:
            _nickname = message.get('message', '')
            _code = message.get('prediction', None)
            _reason = ''
            _detail = {}

            if _code is None:

                _rslt = self.nickname_filter.think(_nickname, detail=True)
                _code = _rslt['code']
                _reason = _rslt['reason_char']
                _detail = _rslt['detail']

            await sync_to_async(self.nickname_filter.saveNicknameRequestRecord)(nickname=_nickname, status=_code)

            await self.channel_layer.group_send(
                self.group_name_admin_client,
                {
                    'type': 'channel_chat_nickname',
                    'nickname': _nickname,
                    'code': _code,
                    'reason_char': _reason,
                    'detail': _detail
                },
            )
            return None

        elif order_key == self.key_training_start:
            pass
            
        
        await self.send(text_data=json.dumps({'msgid': order_key, 'message': message}))

