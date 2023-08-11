from django.core.management.base import BaseCommand
from django.apps import apps
from dataparser.apps import ExcelParser, MessageParser
from dataparser.classes.xml import XmlParser
from dataparser.jsonparser import JsonParser
from ai.classes.translator_pinyin import translate_by_string
import os, json, re
from datetime import date

class Command(BaseCommand):
    help = "parse the dataset file to json data."

    def add_arguments(self, parser):
        parser.add_argument(
            '-i', dest='input_file', required=True,
            help='the path of excel file.',
        )


    def handle(self, *args, **options):
        input_file = options.get('input_file')
        output_path = './dataset.out.json'

        result_list = []

        try:
            print('Start Handle.')
            _xp = XmlParser(file=input_file)
            
            print('Done')

            # _jp = JsonParser(file=output_path)
            # _jp.save(result_list)
            
        except Exception as err:
            print('Parse Dataset Failed.')
            print(err)


        