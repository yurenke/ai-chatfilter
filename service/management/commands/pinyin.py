from django.core.management.base import BaseCommand
from pypinyin import pinyin, Style
from ai.classes.translator_pinyin import translate_by_string
import os

class Command(BaseCommand):
    help = "tracing pinyin for text."

    def add_arguments(self, parser):
        parser.add_argument('-t', dest='text', type=str, help="text")

    def handle(self, *args, **options):
        _text = options.get('text', '')
        self.stdout.write('tracing pinyin :: ' + _text)
        _words = pinyin(_text, strict=False, style=Style.NORMAL, heteronym=True)
        translated = translate_by_string(_text)
        print('pinyin origin words: ', _words)
        print('pinyin translated: ', translated)
    

        