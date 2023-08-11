import re
from datetime import datetime


class ChatStore():
    """

    """
    max_second_reserve = 90
    max_reserve_per_room = 10
    dict_room_lastupdate = {}
    dict_romm_temporary_texts = {}
    temporary_text_list = []
    NO_ROOM_NAME = 'NOROOM'
    max_same_room_word = 2

    def __init__(self):
        self.dict_romm_temporary_texts[self.NO_ROOM_NAME] = []


    def upsert_text(self, text, user, room):
        _now = datetime.now()
        _timestamp = datetime.timestamp(_now)
        _room_name = room if room else self.NO_ROOM_NAME
        _texts = self.dict_romm_temporary_texts.get(_room_name, [])

        _texts = [_ for _ in _texts if _timestamp - _[2] < self.max_second_reserve ][:self.max_reserve_per_room]

        _loc = [text, user, _timestamp]

        _texts.append(_loc)

        self.dict_romm_temporary_texts[_room_name] = _texts

        return self


    def get_texts(self, user, room):
        _room_name = room if room else self.NO_ROOM_NAME
        _texts = self.dict_romm_temporary_texts.get(_room_name, [])
        return _texts

    
    def get_texts_by_room(self, room):
        _room_name = room if room else self.NO_ROOM_NAME
        return [_[0] for _ in self.dict_romm_temporary_texts.get(_room_name, [])]


    def get_merged_text(self, text, user, room):
        texts = self.get_texts(user, room)
        if len(texts) == 0:
            return text

        lastest_text = texts[0]
        res_text = ''

        if len(text) < 5:
            has_general = False
            for _t in text:
                if _t <= u'\u007a':
                    has_general = True
                    break
            
            if has_general:
                res_text = lastest_text[1] + text
            else:
                return text
        else:        
            _now = datetime.now()
            _timestamp = datetime.timestamp(_now)
            if _timestamp - lastest_text[0] < 10:
                res_text = lastest_text[1] + text
            else:
                return text
        
        return res_text if len(res_text) > 0 else None


    def check_same_room_conversation(self, text, room):
        _num_matched = 0
        room_texts = self.get_texts_by_room(room)

        english_text = self.replace_only_left_english(text)
        if english_text:
            if len(english_text) >= 4:
                for _rt in room_texts:
                    if self.replace_only_left_english(_rt) == english_text:
                        return 'not allow same English keep typing'

        
        digital_text = self.replace_only_left_digital(text)
        if digital_text and len(digital_text) >= 3:
            for _rt in room_texts:
                if len(_rt) < 12:
                    _before_digital_text = self.replace_only_left_digital(_rt)
                    if len(_before_digital_text) >= 3:
                        return 'suspect digital merged wechat number'

        # bankerplayer_text = self.replace_only_left_bankerplayer(text)
        # if bankerplayer_text:
        #     for _rt in room_texts:
        #         # if len(_rt) >= 10:
        #         #     continue
        #         if self.replace_only_left_bankerplayer(_rt):
        #             _num_matched += 1
        #             if _num_matched > self.max_same_room_word:
        #                 return 'too many talked on banker and player'

        return ''


    def replace_only_left_bankerplayer(self, _text):
        return re.sub(r'[^(莊庄装閒閑闲贤)]+', '', _text)

    def replace_only_left_english(self, _text):
        return re.sub(r'[^(a-zA-Z)]+', '', _text)

    def replace_only_left_digital(self, _text):
        return re.sub(r'[^(0-9)]+', '', _text)