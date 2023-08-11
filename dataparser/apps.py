from __future__ import absolute_import, division, print_function, unicode_literals

from django.apps import AppConfig

from datetime import datetime

from numpy.core.fromnumeric import product
from .classes.map_hex import mapHexes
from .models import CustomDictionaryWord
from service.models import Blockword
from ai.models import SoundVocabulary, NewVocabulary, DigitalVocabulary, Vocabulary, Language
from ai.classes.translator_pinyin import translate_by_string

import xlrd, openpyxl
import os
import re
import jieba
import pickle


class DataparserConfig(AppConfig):
    name = 'dataparser'




class ExcelParser():
    """
        get_row_list (column=[], limit=0)
        return List
    """
    file = None
    file_content = None
    books = []
    is_multiple = False
    file_extension = re.compile("^(.*).xlsx?$", re.IGNORECASE)

    def __init__(self, **kwargs):

        self.file = kwargs.get(
            'file',
            None,
        )

        self.file_content = kwargs.get(
            'file_content',
            None,
        )

        self.books = []

        if self.file:
            start_time = datetime.now()

            if os.path.isdir(self.file):

                self.is_multiple = True

                for _file in os.listdir(self.file):

                    if self.file_extension.search(_file):
                        if '~' in _file:
                            continue

                        _file_path = '{}/{}'.format(self.file, _file)
                        print('ExcelParser Open File Path: ', _file_path)
                        book = xlrd.open_workbook(_file_path)
                        self.books.append(book)

            else:

                self.is_multiple = False
                book = xlrd.open_workbook(self.file)
                # self.book = book
                self.books.append(book)


            end_time = datetime.now()
            spend_second = (end_time - start_time).total_seconds()

            print('====ExcelParser Loads File spend seconds: ', spend_second)
            print("Worksheet name(s): {0}".format(book.sheet_names()))

        elif self.file_content:

            self.is_multiple = False
            book = xlrd.open_workbook(
                    filename=None,
                    file_contents=self.file_content,
                )
            self.books.append(book)

        else:

            print('ExcelParser Failed With Wrong File path.')


    def get_row_list(self, column=[], limit=0, just_first_sheet=True):
        total_rows = []
        _book_idx = 1
        for book in self.books:

            if just_first_sheet:
                sheet = book.sheet_by_index(0)
                rows = self.get_row_list_by_sheet(sheet, column=column, limit=limit)
                total_rows += rows
                print('Excel Get Row List: Book Index[{}] Row Length: {}'.format(_book_idx, len(rows)))

            _book_idx += 1

        return total_rows
    
    def get_row_list_by_sheet(self, sheet, column=[], limit=0):
        sh = sheet
        # print("==Getting Data in Sheet name: {0}, rows: {1}, cols:{2}".format(sh.name, sh.nrows, sh.ncols))
        ary = []
        _columns = []

        def parse_str(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, int):
                return str(value)
            else:
                return str(int(value))
            
        if len(column) > 0:

            if sh.nrows < limit:
                limit = sh.nrows
            
            for rx in range(limit if limit > 0 else sh.nrows):
                
                child = [x.value for x in sh.row(rx)]
                if rx == 0:
                    for col in column:
                        if type(col) == list:
                            __idx = -1
                            for __c in col:
                                __loc_idx = child.index(__c) if __c in child else -1
                                if __loc_idx >= 0:
                                    __idx = __loc_idx
                                    break
                        else:
                            __idx = child.index(col) if col in child else -1

                        _columns.append(__idx)

                else:
                    
                    _next_child = [parse_str(child[i]) if i >= 0 else '' for i in _columns]
                    ary.append(_next_child)
        else:

            for rx in range(limit if limit > 0 else sh.nrows):

                child = [parse_str(x.value) for x in sh.row(rx)]
                ary.append(child)
        

        return ary


    def export_excel(self, file=None, data=[]):
        _new_book = openpyxl.Workbook()
        _newsheet = _new_book.create_sheet(index=0)

        _r = 1
        for _ in data:
            _c = 1
            for loc in _:
                if isinstance(loc, str) and loc.isdigit():
                    _newsheet.cell(_r, _c).value = int(loc)
                else:
                    _newsheet.cell(_r, _c).value = loc
                _c += 1
            _r += 1

        if file:
            print('Save Trim File: ', file)
            _new_book.save(file)
        


class MessageParser():
    #
    # <table>!TSTCNY20:3</table>
    # 首字母 = # 公開桌 @ 密碼桌 !獨享桌
    # + 用戶名
    # + :3   是桌子序列號
    regex_msg = re.compile("<msg>(.*)</msg>", flags= re.IGNORECASE | re.DOTALL)
    regex_xml_lv = re.compile("<(?:lv|yplv|level)>(\d*?)<\/(?:lv|yplv|level)>", flags= re.IGNORECASE | re.DOTALL)
    regex_xml_anchor = re.compile("<(?:anchor|anchmsg|isAnchorPlatformMsg)>(\w+?)<\/(?:anchor|anchmsg|isAnchorPlatformMsg)>", flags= re.IGNORECASE | re.DOTALL)

    regex_xml_tag = re.compile("<[^>]+?>[A-Za-z0-9\#\@\!\:\_\-]*?</[^>]+?>")
    regex_xml_ignores = re.compile("<(?:at|emj|prm|dt|nick|ye|ya|player|gift|table)>(.*?)<\/(?:at|emj|prm|dt|nick|ye|ya|player|gift|table)>", flags= re.IGNORECASE | re.DOTALL)

    regex_xml_broke_start = re.compile("(<msg>)|(<[\w]+?>)", flags= re.IGNORECASE)
    regex_xml_broke_end = re.compile("<[^>]+>?")
    regex_xml_broke_anchor = re.compile("<(?:anchor|isAnchorPlatformMsg)>(\w+)")

    regex_bracket_lv = re.compile("\{(?:viplv|agviplv)(\d+?)\}", flags= re.IGNORECASE | re.DOTALL)
    regex_bracket_digits = re.compile("\{[a-zA-Z\d\s\:\-]+?\}")

    # regex_xml_clean_at = re.compile("<.?at>", flags= re.IGNORECASE | re.DOTALL)

    def __init__(self):
        print('MessageParser init done.')

    
    def parse(self, string):
        lv = 0
        anchor = 0
        # repres_msg = None
        text = ''
        repres_msg = self.regex_msg.match(string)

        try:

            if repres_msg:

                text = repres_msg.group(1)

                repres_xml_lv = self.regex_xml_lv.search(text)
                if repres_xml_lv:
                    lv = int(repres_xml_lv.group(1))
                    text = self.regex_xml_lv.sub("", text)

                repres_xml_anchor = self.regex_xml_anchor.search(text)
                if repres_xml_anchor:
                    anchor = 0 if repres_xml_anchor.group(1) == '0' else 1
                    text = self.regex_xml_anchor.sub("", text)

                text = self.regex_xml_ignores.sub("", text)
            
            elif self.regex_xml_broke_start.match(string): # broke message..
                text = string.strip()
                repres_xml_lv = self.regex_xml_lv.search(text)

                if repres_xml_lv:
                    lv = int(repres_xml_lv.group(1))
                    text = self.regex_xml_lv.sub("", text)
                
                repres_xml_anchor = self.regex_xml_broke_anchor.search(text)
                if repres_xml_anchor:
                    anchor = 0 if repres_xml_anchor.group(1) == '0' else 1
                    text = self.regex_xml_broke_anchor.sub("", text)
                    
                text = self.regex_xml_tag.sub("", text)
                text = self.regex_xml_broke_start.sub("", text)
                text = self.regex_xml_broke_end.sub("", text)
            else:

                text = string.strip()

                repres_bracket_lv = self.regex_bracket_lv.match(text)

                if repres_bracket_lv:
                    
                    lv = int(repres_bracket_lv.group(1))
                    text = self.regex_bracket_lv.sub("", text)
            

            text = self.regex_bracket_digits.sub("", text)

        except Exception as err:

            text = '[Err○r] {}'.format(err)
        
        # text = self.trim_only_general_and_chinese(text)
        
        # print(text, lv, anchor)

        return text, lv, anchor

    
    def trim_only_general_and_chinese(self, string):
        _result = ''

        for uc in string:
            _code = ord(uc)

            if _code >= 0x4e00 and _code <= 0x9faf:
                # chinese
                _result += uc
                continue
            elif _code == 0x3000 or _code == 0x00a0:
                # full space to half space
                _code = 0x0020
            elif _code > 0xfee0 and _code < 0xffff:
                # full char to half
                _code -= 0xfee0
            
            # 【2000-206F】 General Punctuation 一般標點符號 # 8216
            # if _code >= 0x2000 and _code <= 0x206f: continue
            # 【3000-303F】 CJK Symbols and Punctuation 中日韓符號和標點  # 12290
            # if _code >= 0x3000 and _code <= 0x303f: continue

            if _code == 0x0020 or (_code >= 0x0030 and _code <= 0x0039) or (_code >= 0x0041 and _code <= 0x005a) or (_code >= 0x0061 and _code <= 0x007a):
                _result += chr(_code).lower()

            # if (_code < 0x0030 and _code != 0x0020) or 
            #     _code > 0x7f or 
            #     (_code >= 0x3a and _code <= 0x40 ):
            #     continue
                # out of English or digits
                # _hex16 = hex(_code)
                # _string = mapHexes.get(_code, None)
                
                # if _string:
                #     _result += _string
            # else:
            #     _result += chr(_code).lower()
        
        return _result



class JieBaDictionary():
    """
        
    """
    split_character = '_'
    unknown_character = '#UNK#'
    pad_character = '#PAD#'
    number_character = '#NUM#'
    alphabet_character = '#ALP#'
    reserve_character = '#RES#'
    folder = os.path.dirname(__file__)
    pickle_folder = os.path.dirname(__file__) + '/_pickles'
    freq_json_file = os.path.dirname(__file__) + '/assets/freq.json'
    origin_vocabulary = []
    vocabularies = []
    vocabulary_freqs = []
    none_tone_map = {}
    re_eng = re.compile('[a-zA-Z0-9_]', re.U)
    freq_additional = 750


    def __init__(self, vocabulary=[], freqs=[]):
        self.origin_vocabulary = [self.pad_character, self.unknown_character, self.number_character, self.alphabet_character, self.reserve_character]
        jieba.re_eng = self.re_eng
        jieba.initialize(dictionary=self.folder + '/assets/jieba.txt')

        if not os.path.isdir(self.pickle_folder):
            os.mkdir(self.pickle_folder)

        if len(freqs) > 0:
            assert len(freqs) == len(vocabulary)
            self.load_vocabularies(vocabulary=vocabulary, freqs=freqs)
            self.save_vocabularies()
        else:
            self.load_vocabularies(vocabulary=vocabulary)
        
        print('JieBaDictionary Initial Done.')


    def split_word(self, text='', is_pinyin=False):
        results = []
        unknowns = []
        _buf = ''

        if is_pinyin:

            _list = self.__cut_DAG_NO_HMM(text)
            for _ in _list:
                if not _:
                    continue
                elif  _[-1] == self.split_character:

                    if _buf:
                        __ = _buf + _
                        _buf = ''
                        _none_tone_words = self.get_none_tone_word(__)

                        if _none_tone_words:
                            results += _none_tone_words
                        
                        elif __[:-1].isdigit():
                            results.append(self.number_character)

                        else:
                            if __.count(self.split_character) > 1:
                                __words = __.split(self.split_character)
                                _map = self.none_tone_map
                                for _word in __words:
                                    if _word:
                                        if _map.get(_word, None):
                                            results.append(_word+self.split_character)
                                        elif _word.isdigit():
                                            results.append(self.number_character)
                                        else:
                                            results.append(self.unknown_character)
                                            unknowns.append(_word+self.split_character)
                            else:
                                results.append(self.unknown_character)
                                unknowns.append(__)
                        
                    else:
                        __ = _

                        if len(__) > 1:
                            if __[:-1].isdigit():
                                _none_tone_words = self.get_none_tone_word(__)
                                if _none_tone_words:
                                    results += _none_tone_words 
                                else:
                                    results.append(self.number_character)
                                
                            else:
                                results.append(__)

                else:
                    _buf += _

            if _buf:
                print('split_word [] Why left _buf: {} , text: {}'.format(_buf, text))
                unknowns.append(_buf)
        else:
            _list = jieba.cut(text, HMM=False, cut_all=False) if text else []
            results = [_ for _ in _list]
        
        return results, unknowns


    def __cut_DAG_NO_HMM(self, sentence):
        _DAG = jieba.get_DAG(sentence)
        my_route = self.get_route(sentence, _DAG)
        # print('my_route: ', my_route)
        if len(my_route) > 1:
            _tmp_freq = 0
            _list = []

            for _ in my_route:
                if _['freq'] > _tmp_freq:
                    _tmp_freq = _['freq']
                    _list = _['list']

            # print('[__cut_DAG_NO_HMM] max freq list: ', _list)
            if _list:
                for __ in _list:
                    yield __
            else:
                print('[__cut_DAG_NO_HMM] With No List sentence: {} , route: {}'.format(sentence, my_route))

        else:
            route = {}
            jieba.calc(sentence, _DAG, route)

            x = 0
            N = len(sentence)
            buf = ''
            while x < N:
                y = route[x][1] + 1
                l_word = sentence[x:y]
                if self.re_eng.match(l_word) and len(l_word) == 1:
                    buf += l_word
                    x = y
                else:
                    if buf:
                        yield buf
                        buf = ''
                    yield l_word
                    x = y
            if buf:
                yield buf
                buf = ''


    def get_route(self, sentence, dag):
        _DAG = dag
        _FREQ = jieba.dt.FREQ
        _split_char = self.split_character

        N = len(sentence)

        def new_route():
            return {
                'end': -1,
                'list': [],
                'freq': 0,
            }

        def adj_route(route_list, start_idx):
            dag_list = _DAG[start_idx]
            next_r_list = []
            # print('sentence: {},  start_idx: {},  dag_list: {}'.format(sentence, start_idx, dag_list))

            for _ in route_list:
                _is_route_first = True
                _cloned_list = [__ for __ in _['list']]
                _route_end = _['end']
                _origin_freq = _['freq']
                _route_next_start = _route_end + 1

                if _route_next_start == start_idx:

                    for _end_idx in dag_list:
                        _end_idx_warped = _end_idx
                        while _end_idx_warped < N and sentence[_end_idx_warped] != _split_char:
                            _end_idx_warped += 1

                        _end_idx_warped += 1
                        
                        _sen = sentence[start_idx:_end_idx_warped]
                        _freq_ = _FREQ.get(_sen, 0)
                        _num_splited = _sen.count(_split_char)
                        
                        if _num_splited > 1:
                            _freq_ *= 2**_num_splited
                        else:
                            _freq_ = min(_freq_, 64)
                        # print('=== _sen: {}  _freq_: {}'.format(_sen, _freq_))
                        __next_end = _end_idx_warped - 1

                        if _is_route_first:

                            _['list'].append(_sen)
                            _['end'] = __next_end
                            _['freq'] += _freq_

                            _is_route_first = False

                        else:

                            _new_route = {
                                'end': __next_end,
                                'list': _cloned_list + [_sen],
                                'freq': _origin_freq + _freq_,
                            }
                            
                            next_r_list.append(_new_route)
                            # print('[next_r_list]: ', next_r_list)

                elif _route_next_start < start_idx:
                    print('[WARNING][JiebaDictionary][__cut_DAG_NO_HMM] has gap  : {} ,  start_idx: {}'.format(_, start_idx))
                    # print('          sentence: ', sentence)
                    return []
            
            if next_r_list:
                route_list += next_r_list

            return route_list



        route_list = []
        route_list.append(new_route())

        for _idx in range(N):
            if _idx == 0:
                route_list = adj_route(route_list, _idx)
                continue
            
            _last_idx = _idx-1
            if sentence[_last_idx] == _split_char:  # if this idx is the first character
                route_list = adj_route(route_list, _idx)
        
        return route_list


    def append_vocabularies(self, appended_vocabulary):
        for _av in appended_vocabulary:
            self.add_word(_av, freq=self.freq_additional)

        print('FREQ Length: ', len(jieba.dt.FREQ), ', Vocabulary Length: ', len(self.vocabularies))


    def add_word(self, word, freq = None):
        if self.is_allowed_word(word):
            _is_new_word = self.is_new_word(word)
            _freq = int(freq) if freq else 0
            if _freq < 0:
                _freq = self.freq_additional
            
            if _is_new_word:
                
                self.vocabularies.append(word)
                self.vocabulary_freqs.append(_freq)
                self.add_none_tone_word(word)
                
            else:

                _idx = self.vocabularies.index(word)
                _freq = self.vocabulary_freqs[_idx] + _freq
                self.vocabulary_freqs[_idx] = _freq

            jieba.add_word(word, freq=_freq)

            return True
        return False

    
    def add_none_tone_word(self, word):
        # if word.count(self.split_character) <= 4 and word[-2].isdigit():
        # if word.count(self.split_character) <= 4 and re.match(r'[\d]', word) == None:
        if word.count(self.split_character) <= 4:
            # _no_digit_word = re.sub(r'[\d]+', '', word)
            # _key = _no_digit_word.replace(self.split_character, '')
            _key = word.replace(self.split_character, '')

            words = self.none_tone_map.get(_key, [])
            if word not in words:
                words += [word]
                self.none_tone_map[_key] = words
                return True
        
        return False


    def get_none_tone_word(self, pinyin):
        # print('get_none_tone_word pinyin: ', pinyin)
        g_map = self.none_tone_map
        _pinyin = pinyin.replace(self.split_character, '')
        _basic = g_map.get(_pinyin, None)
        if _basic and len(_basic) == 1:
            return _basic
        # jing_yan_
        _buf = ''
        _tmp_words = []
        _i = 0
        _len = len(_pinyin)
        _words = []
        # merge single word
        while _i < _len:
            _p = _pinyin[_i]
            _buf += _p
            _basic = g_map.get(_buf, None)
            if _basic:
                _tmp_words = _basic
            else:
                if _tmp_words:
                    _words.append(_tmp_words)
                    _tmp_words = []
                    _buf = _p
            _i += 1
        
        _basic = g_map.get(_buf, None)
        if _basic:
            _words.append(_basic)
            # print('get_none_tone_word [] _words: ', _words)
            # merge sentence
            final_words = []
            _i = 0
            _len_words = len(_words)
            _vocabularies = self.vocabularies
            while _i < _len_words:
                _num_find_match = _i
                _ws = _words[_i]
                _next_i = _i +1
                for _w in _ws:
                    _merge_word = _w
                    while _next_i < _len_words:
                        _ws_next = _words[_next_i]
                        _is_find_next_match = False
                        for _w_n in _ws_next:
                            tmp_word = _merge_word + _w_n
                            # _matched = g_map.get(tmp_word, None)
                            _matched = tmp_word in _vocabularies
                            if _matched:
                                _is_find_next_match = True
                                _num_find_match = _next_i
                                _merge_word = tmp_word
                                break
                        
                        if _is_find_next_match:
                            _next_i += 1
                        else:
                            break
                    
                    if _num_find_match > _i:
                        # found match
                        final_words.append(_merge_word)
                        break
                
                if _num_find_match == _i:
                    final_words.append(_ws[0])
                else:
                    _i = _num_find_match
                
                _i += 1

            return final_words
        else:
            # print('word left _buf: ', _buf)
            # print('unknown none tone word _pinyin: ', _pinyin)
            return None


    def get_vocabulary(self, pure = False):
        if pure:
            return self.vocabularies
        else:
            return self.origin_vocabulary + self.vocabularies

    def get_unknown_position(self):
        return self.origin_vocabulary.index(self.unknown_character)

    def get_alphabet_position(self):
        return self.origin_vocabulary.index(self.alphabet_character)

    def get_numeric_position(self):
        return self.origin_vocabulary.index(self.number_character)

    def get_reserve_position(self):
        return self.origin_vocabulary.index(self.reserve_character)

    
    def get_cut_for_search(self, sentence):
        return jieba.cut_for_search(sentence, HMM=False)


    def get_cut_all(self, sentence, min_length = 1):
        _dag = jieba.get_DAG(sentence)
        _n = len(sentence)
        result = []
        
        for _idx in range(_n):
            if _idx == 0 or sentence[_idx-1] == self.split_character:
                _dag_list = _dag[_idx]
                for __x in _dag_list:
                    _word = sentence[_idx:__x+1]
                    if _word.count(self.split_character) >= min_length:
                        result.append(_word)
            
        return result


    def get_DAG(self, sentence):
        return jieba.get_DAG(sentence)


    def get_vocabulary_freq_list(self):
        return self.vocabulary_freqs
    

    def is_new_word(self, _word):
        return _word not in self.vocabularies

    
    def is_allowed_word(self, _word):
        if _word:
            if _word[-1] == self.split_character:
                return len(_word) > 1
            else:
                for _ in _word:
                    if _ < u'\u4e00' or _ > u'\u9fa5':
                        return False
                return True
        return  False

    
    def load_vocabularies(self, vocabulary=None, freqs=None):
        _list = []
        _freqs = freqs
        if isinstance(vocabulary, list) and len(vocabulary) >0:
            _list = vocabulary

            if _freqs and len(_list) == len(_freqs):
                print('[Jieba][Load_Vocabularies] by Vocabulary Data Length: ', len(vocabulary), flush=True)
                print('    Beginning Data: ',  vocabulary[:10], flush=True)
                print('    Ending Data: ',  vocabulary[-10:], _freqs[-10:], flush=True)
            else:
                _freqs = [2 for _ in _list]

        else:

            with open(self.pickle_folder + '/tokenizer_vocabularies.pickle', 'rb') as handle:
                _pickled = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)
                _list = _pickled[0]
                _freqs = _pickled[1]
        

        for _idx, _ in enumerate(_list):
            # self.add_word(_, freq=_freq)
            _freq = _freqs[_idx]
            self.add_word(_, freq=_freq)


    def save_vocabularies(self):
        _full_data = [self.vocabularies, self.vocabulary_freqs]
        
        with open(self.pickle_folder + '/tokenizer_vocabularies.pickle', 'wb+') as handle:
            pickle.dump(_full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)




class EnglishParser():
    """

    """
    _vocabularies = []
    _regex_english = r'[a-zA-Z]+'
    _regex_reversed_english = re.compile("[^(a-z)\s]+", re.IGNORECASE)

    def __init__(self, vocabularies = []):

        if vocabularies:
            self.set_vocabulary(vocabularies)


    def set_vocabulary(self, vocabularies = []):
        if vocabularies:
            
            if isinstance(vocabularies[0], str):

                self._vocabularies = vocabularies
            
            else:

                self._vocabularies = [v[0] for v in vocabularies]

        else:

            self._vocabularies = self.get_vocabulary_by_database()
            print('[EnglishParser] Get Vocabulary On Database.')
            
        print('[EnglishParser][set_vocabulary] _vocabularies Length: ', len(self._vocabularies))

    
    def get_vocabulary(self):
        return self._vocabularies


    def get_vocabulary_by_database(self):
        _language = Language.objects.get(code='EN')
        _vocabulary = list(Vocabulary.objects.filter(language=_language).values_list('context', flat=True))
        return _vocabulary


    def replace_to_origin_english(self, text):
        
        
        new_text = re.sub(self._regex_english, self._sub_match_fn, text)
            
        # print('new_text: ', new_text)

        return new_text

    def _sub_match_fn(self, match):
        _diversity_suffix = ['ves', 'ies', 'es', 's', 'ing', 'ed']
        match_str = match.group()
        # print('match_str: ', match_str)
        if match_str not in self._vocabularies:
            for _dss in _diversity_suffix:
                _suffix = '{}$'.format(_dss)
                if re.findall(_suffix, match_str):
                    _fixed_str = re.sub(_suffix, '', match_str)
                    # print('_fixed_str: ', _fixed_str)

                    if _dss == 'ves':
                        _fixed_str_example = _fixed_str + 'f'
                        if _fixed_str_example in self._vocabularies:
                            return _fixed_str_example
                        _fixed_str_example = _fixed_str + 'fe'
                        if _fixed_str_example in self._vocabularies:
                            return _fixed_str_example

                    elif _dss == 'ies':
                        _fixed_str_example = _fixed_str + 'y'
                        if _fixed_str_example in self._vocabularies:
                            return _fixed_str_example

                    if _fixed_str in self._vocabularies:
                        return _fixed_str

        return match_str


    def replace_to_only_english(self, text):
        return re.sub(r'[^(a-zA-Z\s)]+', '', text)


    def parse_right_vocabulary_list(self, eng_str):
        result_list = []
        _eng_string = eng_str.strip().lower()
        _eng_list = re.split(r'\s+', _eng_string)

        for _ in _eng_list:
            if re.match(self._regex_english, _):
                if _ not in self._vocabularies:
                    return []
                else:
                    result_list.append(_)
            else:
                return []

        return result_list

    def trim(self, text):
        return re.sub(self._regex_reversed_english, '', text).strip()

    def is_vocabulary(self, text):
        return text in self._vocabularies

