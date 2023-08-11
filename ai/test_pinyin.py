import os
from pypinyin import pinyin, Style

if __name__ == '__main__':

    while True:
        print('Please Enter..')
        _input = input()

        if not _input or _input == 'exit':
            break
        
        _words = pinyin(_input, style=Style.NORMAL, heteronym=True)
        print('words: ', _words)

        _words_first = [_w[0] for _w in _words]
        print('words first: ', _words_first)

        _next = ''.join(_words_first)
        print('result: ', _next)