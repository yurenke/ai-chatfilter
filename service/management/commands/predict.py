from django.core.management.base import BaseCommand

from ai.predict import predict_by_ai, predict_by_excel_file
import os, json, xlwt
# import gc


class Command(BaseCommand):
    help = 'predict text status possible'

    def add_arguments(self, parser):
        parser.add_argument(
            '-t', dest='text', required=False,
            help='the text to be predicted.',
        )
        parser.add_argument(
            '-i', dest='input_file', required=False,
            help='the file of need to be predicted.',
        )
        parser.add_argument(
            '-s', dest='silence', required=False, action='store_true',
            help='silence mode.',
        )
        parser.add_argument(
            '-plus', dest='plus', required=False, action='store_true',
            help='plus mode.',
        )
        parser.add_argument(
            '-json', dest='output_json', required=False, action='store_true',
            help='whether output the result to json file.',
        )

        parser.add_argument(
            '-excel', dest='output_excel', required=False, action='store_true',
            help='whether output the result to excel file.',
        )


    def handle(self, *args, **options):
        text = options.get('text', None)
        input_file = options.get('input_file', None)
        silence = options.get('silence', False)
        plus = options.get('plus', False)
        output_json = options.get('output_json', False)
        output_excel = options.get('output_excel', False)

        if text:

            self.stdout.write('Word of Prediction: ' + text)
            res = predict_by_ai(text, silence=silence, detail=True)
            self.stdout.write('Prediction is: ' + str(res))

        elif input_file:

            full_file_path = os.getcwd() + '/' + input_file
            self.stdout.write('Full input excel file path: ' + full_file_path)

            ratio, book = predict_by_excel_file(
                file=full_file_path,
                silence=silence,
                plus=plus,
                output_json=output_json,
                output_excel=output_excel,
            )

            print('textbook size: ', len(book))

        else:
            
            self.stdout.write('Nothing happend.')

        self.stdout.flush()

    
        

        

        
        

        