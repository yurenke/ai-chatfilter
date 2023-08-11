from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from datetime import datetime
from service.models import Blockword
from zhconv import convert

class FuzzyCenter():
    """
    """
    wording_define_ratio = 70

    block_words = []
    block_sentence = []
    


    def __init__(self):
        self.refresh_block_words()
        

    
    def refresh_block_words(self):
        blockword_objects = Blockword.objects.all()
        block_words = []
        block_sentence = []
        for blockword in blockword_objects:
            
            _text = blockword.text
            _cn, _zh = self.parse_cnzh(_text)

            if len(_text) == 1:
                if not _text in block_words:
                    block_words.append(_text)
                    if _cn != _text:
                        block_words.append(_cn)
                    elif _zh != _text:
                        block_words.append(_zh)
            else:
                if not _text in block_sentence:
                    block_sentence.append(_text)
                    if _cn != _text:
                        block_sentence.append(_cn)
                    elif _zh != _text:
                        block_sentence.append(_zh)

        self.block_words = block_words
        self.block_sentence = block_sentence
        return self

    
    def parse_cnzh(self, txt):
        _cn = convert(txt, 'zh-cn')
        _zh = convert(txt, 'zh-tw')
        return _cn, _zh



