from django.core.management.base import BaseCommand
import os

class Command(BaseCommand):
    help = "seed database for testing and development."

    def add_arguments(self, parser):
        parser.add_argument('--mode', type=str, help="Mode")

    def handle(self, *args, **options):
        self.stdout.write('seeding data...')
        self.run_seed(self, options['mode'])
        self.stdout.write('done.')

    def run_seed(self, mode):
        """ Seed database based on mode

        :param mode: refresh / clear 
        :return:
        """
        # Clear data from tables
        # clear_data()
        if mode == MODE_CLEAR:
            return

        # Creating 15 addresses
        # for i in range(15):
        #     create_address()

        