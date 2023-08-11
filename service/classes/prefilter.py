import re
import time
from ai.classes.translator_pinyin import translate_by_string


regex_chinese = re.compile('[\u4e00-\u9fa5]+')
regex_korea = re.compile('[\uac00-\ud7a3]+')
allowed_character_regexies = [
    (u'\u0020', u'\u0082'), # general english, digits and symbol
    # (u'\u23e9', u'\u23f9'), # symbol
    # (u'\u26bd', u'\u270d'), # symbol
    (u'\u3001', u'\u3002'), # dot symbol
    # (u'\u3105', u'\u3129'), # zuyin
    (u'\u4e00', u'\u9fa5'), # chinese
    # (u'\u3041', u'\u30ff'), # japanese
    # (u'\u1100', u'\u11f9'), # korea yin
    # (u'\u3131', u'\u318e'), # korea yin 2
    # (u'\uac00', u'\ud7a3'), # korea
    (u'\uff01', u'\uff65'), # full type of english, digits and symbol
    # (u'\U0001f600', u'\U0001f64f'), # faces
    # (u'\U0001f910', u'\U0001f9ff'), # faces
]

number_pinyin_set = set([
    'yi',
    'er', 'e',
    'san', 'shan',
    'si', 'shi',
    'wu', 
    'liu',
    'qi',
    'ba',
    'jiu', 
    'ling', 'lin'
])

allowed_number_units_chinese_chars = set(['十', '拾', '百', '千', '万', '亿'])



class PreFilter():
    """
    
    """
    SECOND_QUICK_SAYING = 3
    map_loginname_timestamp = {}
    dynamic_pinyin_block_list = []
    dynamic_alert_words_list = []
    dynamic_nickname_pinyin_block_list = []


    def __init__(self):
        pass



    def find_not_allowed_chat(self, text):
        next_char = ''

        text = self.replace_face_symbol(text)
        for u in text:
            if not self.is_allowed_character(u):
                next_char += u

        return next_char

    
    def find_korea_mixed(self, text):
        _korea_words = ''
        _korea_map = {}
        for u in text:
            if regex_korea.match(u):
                _korea_words += u
                if _korea_map.get(u):
                    _korea_map[u] += 1
                else:
                    _korea_map[u] = 1
        
        _length_korea = len(_korea_words)
        _length_text = len(text)
        _ratio = _length_korea / _length_text
        if _ratio == 1:
            _counting_double = 0
            for _kchar in _korea_map:
                if _korea_map[_kchar] > 1:
                    _counting_double += _korea_map[_kchar]

            _ratio_double = _counting_double / _length_korea
            if _ratio_double > 0.25:
                return _korea_words

        elif _length_korea >= 3:
            return _korea_words

        return ''

    
    def find_wechat_char(self, text, lowercase_only = True):
        number_size = 0
        eng_size = 0
        next_char = ''
        _idx = 0
        _last_number_idx = 0
        _dobule_scored_number_range = 2
        _text_ = text.replace(' ', '')
        _has_wv = False

        if lowercase_only:
            _text_ = _text_.lower()
        
        length_char = len(_text_)

        if length_char == 0:
            return ''
    
        for u in _text_:
            _idx += 1
            if self.is_number(u):
                if '0' in next_char and u == '0':
                    continue
                _number_scored = 1
                if number_size > 0 and (_idx - _last_number_idx) <= _dobule_scored_number_range and u != next_char[-1]:
                    _number_scored += 1
                
                number_size += _number_scored
                _last_number_idx = _idx
            elif self.is_english(u):
                eng_size += 1
                if u in 'vVwW' or u >= u'\uff36':
                    _has_wv = True
            else:
                continue

            next_char += u

        _NE_size = number_size + eng_size

        _NE_ratio = _NE_size / length_char
        
        is_many_asci = _NE_size >= 6 or number_size > 4

        is_many_language = _NE_size >= 5 and _NE_ratio > 0.3 and _NE_ratio < 1 and (number_size > 0 or eng_size > 0)

        has_double_eng = False
        has_doubt_eng = False
        if _NE_ratio > 0.8 and eng_size > 5:
            if eng_size <= 8:
                has_doubt_eng = True
            else:
                __first_char = text[:2]
                __next_same_char = 0
                for __idx in range(len(text)):
                    if __idx > 1 and text[__idx: __idx+2] == __first_char:
                        __next_same_char = __idx
                        break
                
                __first_sentence = text[:__next_same_char]
                
                if __next_same_char > 0 and len(__first_sentence) < 12:
                    __left_text = text[__next_same_char:]

                    if __first_sentence in __left_text:
                        has_double_eng = True
        
        # all is english and digits
        if _NE_ratio == 1 and number_size > 0 and length_char > 3 and length_char <= 12:
           return next_char

        # print('[Prefilter][find_wechat_chat] _text_:', _text_)
        # print('[Prefilter][find_wechat_chat] _has_wv:', _has_wv)
        # print('[Prefilter][find_wechat_chat] eng_size:', eng_size)
        # print('[Prefilter][find_wechat_chat] number_size:', number_size)
        if _has_wv and eng_size < 3 and (length_char - eng_size) > 1:
            return next_char

        return next_char if is_many_asci or is_many_language or has_double_eng or has_doubt_eng else ''

    
    def find_emoji_word_mixed(self, text):
        _r_emoji = r'\{\d{1,3}\}'
        _has_emoji = re.search(_r_emoji, text)
        if _has_emoji:
            _pured = re.sub(_r_emoji, '', text).strip()
            if len(_pured) > 0:
                return '#emoji#'
        return ''

    
    def find_unallow_eng(self, text):
        # 
        _unallow_engs = ['wei']
        for _ in _unallow_engs:
            if _ in text:
                return _
        return ''

    def find_unallow_nickname_eng(self, text):
        _unallow_engs = ['wei', 'ag', 'bbin']
        for _ in _unallow_engs:
            if _ in text:
                return _
        return ''


    def find_suspect_digits_symbol(self, text):
        regex = r'[\（\）\！\!\(\)\d\&\+\＋]'
        searched = re.findall(regex, text)
        if searched and len(searched) > 3:
            return ''.join(searched)
        regex_2 = r'[\+\＋\&\＆]'
        searched = re.findall(regex_2, text)
        if len(searched) > 0 and len(re.sub(regex_2, '', text)) >= 3:
            return ''.join(searched)
        return ''



    def is_chinese(self, uchar):
        return (uchar >= u'\u4e00' and uchar <= u'\u9fa5') or (uchar >= u'\uf970' and uchar <= u'\ufa6d')

    def is_zuyin(self, uchar):
        return uchar >= u'\u3105' and uchar <= u'\u312b'

    def is_general(self, uchar):
        return uchar >= u'\u0020' and uchar <= u'\u009b'

    def is_japan(self, uchar):
        return uchar >= u'\u3040' and uchar <= u'\u33ff'

    def is_full_character(self, uchar):
        #  _code > 0xfe00 and _code < 0xffff:
        # return (uchar >= u'\uff00' and uchar <= u'\uff65') or (uchar >= u'\ufe30' and uchar <= u'\ufe6a')
        return uchar >= u'\ufe30' and uchar <= u'\uff65'

    def is_number(self, uchar, chinese=False):
        # return (uchar >= u'\u0030' and uchar <= u'\u0039') or (uchar >= u'\uff10' and uchar <= u'\uff19')
        chineses = [
            u'\u4e00', u'\u58f9', u'\u4e8c', u'\u8cb3', u'\u4e09', u'\u53c1', u'\u56db', u'\u8086', u'\u4e94', u'\u4f0d', 
            u'\u5348', u'\u821e', u'\u516d', u'\u9678', u'\u4e03', u'\u67d2', u'\u516b', u'\u5df4', u'\u53ed', u'\u634c', 
            u'\u6252', u'\u4e5d', u'\u4e45', u'\u7396', u'\u9152', u'\u96f6', u'\u9748', '＋', '+', '扒', '凌', '陵', '仁',
            '灵', '漆', '舞', '武', '医', '陆', '司', '久', '删', '酒', '林', '腰', '兰', '溜', '临', '寺', '期', '铃',
            '山', '遛', '思', '妖', '贰', '玲', '午', '妻', '跋', '衣', '似', '伶', '疤', '韭', '镹', '聆', '易', '衫', '齐',
            '死', '世', '芭', '令', '依', '市', '士', '伊', '柳', '斯', '珊', '流', '奇', '数', '趴', '灸', '凄', '耙',
            '两', '留', '耳', '儿', '羚', '鈴', '义', '旧', '帕', '兒', '霸', '韭', '琳', '双', '俩', '爸', '龄', '乙',
            '究', '耀', '拔', '邻', '恶', '而', '姍', '试', '伤', '叄', '澪', '無', '麟', '式', '舅', '臼', '吾', '辆',
            '无', '撕', '噩', '琦', '琪', '洞', '亿', '柿', '侍', '丸', '琉', '厄', '兔', '訕', '倆', '伺', '骑', '棋',
            '巴', '仇', '杂', '怡', '丝', '棱', '仪', '欺', '&' , '+', '仙', '疚', '夭', '寺', '鸠', '楞', '柺',
        ]
        if chinese:
            return uchar in chineses
        return (uchar >= u'\u0030' and uchar <= u'\u0039') or uchar in chineses

    def is_number_pinyin(self, uchar):
        if u'\u4e00' <= uchar and uchar <= u'\u9fa5' and uchar not in allowed_number_units_chinese_chars:
            _pinyin = self.parse_pinyin(uchar)
            return _pinyin.split('_')[0] in number_pinyin_set
        return False

    def is_english(self, uchar):
        # return (uchar >= u'\u0041' and uchar <= u'\u0039') or (uchar >= u'\u0061' and uchar <= u'\u007a')
        return (uchar >= u'\u0061' and uchar <= u'\u007a') or uchar >= u'\uff21'

    def is_question_mark(self, uchar):
        return uchar == u'\u003f'


    def is_allowed_character(self, uchar):
        for _ in allowed_character_regexies:
            _st = _[0]
            _ed = _[1]
            if uchar <= _ed:
                if uchar >= _st:
                    return True
                break
        return False


    def replace_face_symbol(self, _text):
        regexs = [
            r'°□°',
            r'＾∀＾',
            r'✿‿✿',
            r'´◡`',
            r'༼ つ ◕_◕ ༽つ',
            r'◕_◕',
            r'✪▽✪',
            r'(ง •̀_•́)ง',
            r'(•̀⌄•́)',
            r'ฅ՞•ﻌ•՞ฅ',
            r'(◔.̮◔)',
            r'(ღ♡‿♡ღ)',
            r'(○^㉨^)',
            r'≧∇≦',
            r'￣▽￣',
        ]
        for _r in regexs:
            _text = re.sub(_r, '', _text)
        return _text


    def check_loginname_shorttime_saying(self, loginname=''):
        if loginname:
            _now = time.time()
            _before = self.map_loginname_timestamp.get(loginname, 0)
            self.map_loginname_timestamp[loginname] = _now
            if _before > 0:
                return _now - _before < self.SECOND_QUICK_SAYING
        return False

    def find_alert_words(self, text):
        _text = re.sub(r'[\d\s]+', '', text)

        for _aw in self.dynamic_alert_words_list:
            if _aw in _text:
                return _aw
        return False

    def reg_pinyin_check(self, _pstring):
        must_block_pinyin = [['gong_', 'zhong_'], ['gong_', 'zong_'], ['pin_', 'yin_'], ['ping_', 'yin_'], ['pin_', 'ying_'], ['ping_', 'ying_']]
        
        for words in must_block_pinyin:
            pattern = r'.*'
            for w in words:
                pattern = pattern + w + r'.*'
            rslt = re.search(pattern, _pstring)
            if rslt:
                return ' '.join(words)            
        
        return ''

    def find_pinyin_blocked(self, text):
        _pinyin = self.parse_pinyin(text)
        reg_check_rslt = self.reg_pinyin_check(_pinyin)
        if reg_check_rslt:
            return reg_check_rslt
        # print('[find_pinyin_blocked] translate_by_string _pinyin: ',  _pinyin)
        for _py in self.dynamic_pinyin_block_list:
            if _py in _pinyin:
                # print('find_pinyin_blocked : ', _py)
                return _py
        return False

    def find_nickname_pinyin_blocked(self, text):
        _pinyin = self.parse_pinyin(text)
        # reg_check_rslt = self.reg_pinyin_check(_pinyin)
        # if reg_check_rslt:
        #     return reg_check_rslt

        frequent_block_list = ['ya_you_', 'ya_bo_', 'wan_bo_', 'yun_ding_', 'xin_pu_jing_', 'zun_xiang_hui_', 'hai_na_', 'yu_le_cheng_']
        check_list = frequent_block_list + self.dynamic_nickname_pinyin_block_list
        # print('[find_pinyin_blocked] translate_by_string _pinyin: ',  _pinyin)
        for _py in check_list:
            if _py in _pinyin:
                # print('find_pinyin_blocked : ', _py)
                return _py
        return False

    def parse_pinyin(self, text):
        # _parsed_text = re.sub(r'[a-zA-Z\d\s]+', '', text)
        _parsed_text = re.sub(r'[\d\s]+', '', text)
        _parsed_text = self.parse_riddle(_parsed_text)
        return translate_by_string(_parsed_text)

    def parse_riddle(self, text):
        _nagative_words = ['不要', '拿走', '删除', '后面']
        _idx_nagative_word = 0
        _len_nagative_word = 0
        _check_text = text[3:]
        if _check_text:
            for _nw in _nagative_words:
                if _nw in _check_text:
                    _idx_nagative_word = text.index(_nw)
                    _len_nagative_word = len(_nw)
                    break
        
            if _idx_nagative_word > 0:
                _idx_splited = _idx_nagative_word + _len_nagative_word
                _before_word = text[:_idx_splited]
                _after_word = text[_idx_splited:]
                for _aw in _after_word:
                    if _aw in _before_word:
                        _before_word = _before_word.replace(_aw, '')

                return _before_word + _after_word

        return text


    def set_pinyin_block_list(self, _list_):
        print('[set_pinyin_block_list] Length: ', len(_list_))
        _pinyin_idx = 2
        if len(_list_) > 0:
            if isinstance(_list_[0], str):
                self.dynamic_pinyin_block_list = _list_
            else:
                try:
                    self.dynamic_pinyin_block_list = [_[_pinyin_idx] for _ in _list_]
                except Exception as err:
                    print('[set_pinyin_block_list] Error: ', err)
                    return False
        else:
            self.dynamic_pinyin_block_list = []

        return True

    def set_nickname_pinyin_block_list(self, _list_):
        print('[set_nickname_pinyin_block_list] Length: ', len(_list_))
        _pinyin_idx = 2
        if len(_list_) > 0:
            if isinstance(_list_[0], str):
                self.dynamic_nickname_pinyin_block_list = _list_
            else:
                try:
                    self.dynamic_nickname_pinyin_block_list = [_[_pinyin_idx] for _ in _list_]
                except Exception as err:
                    print('[set_nickname_pinyin_block_list] Error: ', err)
                    return False
        else:
            self.dynamic_nickname_pinyin_block_list = []

        return True

    
    def set_alert_words_list(self, _list_):
        print('[set_alert_words_list] Length: ', len(_list_))
        _word_idx = 1
        if len(_list_) > 0:
            if isinstance(_list_[0], str):
                self.dynamic_alert_words_list = _list_
            else:
                try:
                    self.dynamic_alert_words_list = [_[_word_idx] for _ in _list_]
                except Exception as err:
                    print('[set_alert_words_list] Error: ', err)
                    return False
        else:
            self.dynamic_alert_words_list = []

        return True    


    def get_pinyin_block_list(self):
        return self.dynamic_pinyin_block_list

    
    def get_alert_words_list(self):
        return self.dynamic_alert_words_list

    def get_nickname_pinyin_block_list(self):
        return self.dynamic_nickname_pinyin_block_list
    