from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import DEFAULT_DB_ALIAS, connections, transaction, DatabaseError
import os, time
from datetime import datetime


class Command(BaseCommand):
    help = "backup training data for chat and nickname models."

    # def add_arguments(self, parser):
        # parser.add_argument(
        #     '-m', dest='model_name', required=True,
        #     help='the name of model.',
        # )
        # parser.add_argument(
        #     '-a', dest='app_name', required=False,
        #     help='the name of app.',
        # )

    def handle(self, *args, **options):
        _st_time = datetime.now()
        # model_name = options.get('model_name', None)
        # app_name = options.get('app_name', 'service')
        ModelChatTrainingData = apps.get_model(app_label='ai', model_name='TextbookSentense')
        ModelNicknameTrainingData = apps.get_model(app_label='ai', model_name='NicknameTextbook')

        _table_name_chat_training_data = ModelChatTrainingData._meta.db_table
        _table_name_nickname_training_data = ModelNicknameTrainingData._meta.db_table
        _end_date = (str(_st_time).split(' ')[0]).replace('-', '_')

        _backup_table_name_chat_training_data = '{}_{}'.format(_table_name_chat_training_data, _end_date)
        _backup_table_name_nickname_training_data = '{}_{}'.format(_table_name_nickname_training_data, _end_date)
        
        _sql_backup_training_data = "CREATE TABLE {} AS TABLE {};"
        _sql_backup_chat = _sql_backup_training_data.format(_backup_table_name_chat_training_data, _table_name_chat_training_data)
        _sql_backup_nickname = _sql_backup_training_data.format(_backup_table_name_nickname_training_data, _table_name_nickname_training_data)

        # _sql_tmp_training_data = "CREATE TABLE {} (LIKE {} INCLUDING ALL);"
        # _sql_chat_create_tmp = _sql_tmp_training_data.format(_backup_table_name_chat_training_data, _table_name_chat_training_data)
        # _sql_nickname_create_tmp = _sql_tmp_training_data.format(_backup_table_name_nickname_training_data, _table_name_nickname_training_data)
        _sql_truncate_original = "TRUNCATE TABLE {};"
        _sql_truncate_chat = _sql_truncate_original.format(_table_name_chat_training_data)
        _sql_truncate_nickname = _sql_truncate_original.format(_table_name_nickname_training_data)

        _sql_chat_keep_latest_from_backup = """
            INSERT INTO {} (id, origin, text, weight, status, keypoint, reason)
            SELECT
                ROW_NUMBER() OVER (ORDER BY id ASC) AS id,
                origin,
                text,
                weight,
                status,
                keypoint,
                reason
            FROM (
                SELECT *
                FROM {}
                ORDER BY id DESC
                LIMIT 800000
            ) sub
            ORDER BY id ASC;
        """.format(_table_name_chat_training_data, _backup_table_name_chat_training_data)

        _sql_nickname_keep_latest_from_backup = """
            INSERT INTO {} (id, origin, text, status)
            SELECT
                ROW_NUMBER() OVER (ORDER BY id ASC) AS id,
                origin,
                text,
                status
            FROM (
                SELECT *
                FROM {}
                ORDER BY id DESC
                LIMIT 800000
            ) sub
            ORDER BY id ASC;
        """.format(_table_name_nickname_training_data, _backup_table_name_nickname_training_data)

        _sql_set_id_seq = "SELECT setval('{}_id_seq', (SELECT MAX(id) FROM {}) + 1, false);"
        _sql_set_chat_id_seq = _sql_set_id_seq.format(_table_name_chat_training_data, _table_name_chat_training_data)
        _sql_set_nickname_id_seq = _sql_set_id_seq.format(_table_name_nickname_training_data, _table_name_nickname_training_data)

        _sql_find_tables = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_name SIMILAR TO '({}|{})_%';
                    """
        _sql_find_tables_to_export = _sql_find_tables.format(_table_name_chat_training_data, _table_name_nickname_training_data)
        
        connection = connections[DEFAULT_DB_ALIAS]

        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    print("Executing database operation...")
                    cursor.execute(_sql_backup_chat)
                    cursor.execute(_sql_truncate_chat)
                    cursor.execute(_sql_chat_keep_latest_from_backup)
                    cursor.execute(_sql_set_chat_id_seq)

                    cursor.execute(_sql_backup_nickname)
                    cursor.execute(_sql_truncate_nickname)
                    cursor.execute(_sql_nickname_keep_latest_from_backup)
                    cursor.execute(_sql_set_nickname_id_seq)

                    cursor.execute(_sql_find_tables_to_export)
                    tables = cursor.fetchall()

                    for (tablename,) in tables:
                        # 生成並執行COPY命令，將表內容導出到CSV
                        copy_query = f"COPY {tablename} TO '/tmp/{tablename}.csv' CSV HEADER;"
                        cursor.execute(copy_query)

                        # 生成並執行DROP命令，刪除該表
                        drop_query = f"DROP TABLE {tablename} CASCADE;"
                        cursor.execute(drop_query)

        except DatabaseError as e:
            # 捕捉到資料庫相關錯誤時，進行回滾
            print(f"Transaction failed due to a database error: {e}")

        except Exception as e:
            # 捕捉到其他一般異常時，進行回滾
            print(f"Transaction failed due to an unexpected error: {e}")
        
        else:
            # 如果沒有引發異常，則表示交易成功
            print("Transaction completed successfully without any errors.")

            print('Done Backup Table [{}] ==> [{}]'.format(_table_name_chat_training_data, _backup_table_name_chat_training_data))
            print('Done Backup Table [{}] ==> [{}]'.format(_table_name_nickname_training_data, _backup_table_name_nickname_training_data))
            _ed_time = datetime.now()
            print('Spend Time: ', _ed_time - _st_time)


        