from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import DEFAULT_DB_ALIAS, connections, transaction, DatabaseError
import os, time
from datetime import datetime


class Command(BaseCommand):
    help = "backup data in database."

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
        ModelGoodSentence = apps.get_model(app_label='service', model_name='GoodSentence')
        ModelBlockedSentence = apps.get_model(app_label='service', model_name='BlockedSentence')
        ModelChangeNicknameRequest = apps.get_model(app_label='service', model_name='ChangeNicknameRequest')
        _first_row = ModelGoodSentence.objects.first()

        if not _first_row:
            print('Nothing To Backup.')
            exit(2)

        _table_name = ModelGoodSentence._meta.db_table
        _table_name_blocked = ModelBlockedSentence._meta.db_table
        _table_name_changenickname = ModelChangeNicknameRequest._meta.db_table
        _st_date = (str(_first_row.date).split(' ')[0]).replace('-', '_')
        _end_date = (str(_st_time).split(' ')[0]).replace('-', '_')

        _new_table_name = '{}_{}_to_{}'.format(_table_name, _st_date, _end_date)
        _new_table_name_blocked = '{}_{}_to_{}'.format(_table_name_blocked, _st_date, _end_date)
        _new_table_name_changenickname = '{}_{}_to_{}'.format(_table_name_changenickname, _st_date, _end_date)
        
        print("New Table Name: ", _new_table_name)
        _sql_rename_table = 'ALTER TABLE {} RENAME TO {};'.format(_table_name, _new_table_name)
        _sql_rename_table_2 = 'ALTER TABLE {} RENAME TO {};'.format(_table_name_blocked, _new_table_name_blocked)
        _sql_rename_table_3 = 'ALTER TABLE {} RENAME TO {};'.format(_table_name_changenickname, _new_table_name_changenickname)

        _sql_create_string = 'CREATE TABLE {} AS SELECT * FROM {} WHERE FALSE;'
        _sql_create_new_table = _sql_create_string.format(_table_name, _new_table_name)
        _sql_create_new_table_2 = _sql_create_string.format(_table_name_blocked, _new_table_name_blocked)
        _sql_create_new_table_3 = _sql_create_string.format(_table_name_changenickname, _new_table_name_changenickname)

        _sql_alter_str = 'ALTER TABLE {} ADD PRIMARY KEY(id);'
        _sql_alter_id = _sql_alter_str.format(_table_name)
        _sql_alter_id_2 = _sql_alter_str.format(_table_name_blocked)
        _sql_alter_id_3 = _sql_alter_str.format(_table_name_changenickname)

        _sql_restart_id_str = 'ALTER SEQUENCE {}_id_seq RESTART WITH 1;'
        _sql_restart_id = _sql_restart_id_str.format(_table_name)
        _sql_restart_id_2 = _sql_restart_id_str.format(_table_name_blocked)
        _sql_restart_id_3 = _sql_restart_id_str.format(_table_name_changenickname)

        _sql_alter_str_autoincrement = "ALTER TABLE {} ALTER COLUMN id SET DEFAULT nextval('{}_id_seq');"
        _sql_alter_auto = _sql_alter_str_autoincrement.format(_table_name, _table_name)
        _sql_alter_auto_2 = _sql_alter_str_autoincrement.format(_table_name_blocked, _table_name_blocked)
        _sql_alter_auto_3 = _sql_alter_str_autoincrement.format(_table_name_changenickname, _table_name_changenickname)
        
        _sql_find_tables = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_name SIMILAR TO '({}|{}|{})_%';
                    """
        _sql_find_tables_to_export = _sql_find_tables.format(_table_name, _table_name_blocked, _table_name_changenickname)

        # print('SQL1 : {}'.format(_sql_rename_table))
        # print('SQL2 : {}'.format(_sql_create_new_table))
        
        connection = connections[DEFAULT_DB_ALIAS]
        
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    if _st_date == _end_date:
                        cursor.execute('DROP TABLE IF EXISTS {};'.format(_new_table_name))
                        cursor.execute('DROP TABLE IF EXISTS {};'.format(_new_table_name_blocked))
                        cursor.execute('DROP TABLE IF EXISTS {};'.format(_new_table_name_changenickname))
                    
                    cursor.execute(_sql_rename_table)
                    cursor.execute(_sql_create_new_table)
                    cursor.execute(_sql_alter_id)
                    cursor.execute(_sql_restart_id)
                    cursor.execute(_sql_alter_auto)
                    cursor.execute(_sql_rename_table_2)
                    cursor.execute(_sql_create_new_table_2)
                    cursor.execute(_sql_alter_id_2)
                    cursor.execute(_sql_restart_id_2)
                    cursor.execute(_sql_alter_auto_2)
                    cursor.execute(_sql_rename_table_3)
                    cursor.execute(_sql_create_new_table_3)
                    cursor.execute(_sql_alter_id_3)
                    cursor.execute(_sql_restart_id_3)
                    cursor.execute(_sql_alter_auto_3)

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

            print('Done Rename Table [{}] ==> [{}]'.format(_table_name, _new_table_name))
            print('Done Rename Table [{}] ==> [{}]'.format(_table_name_blocked, _new_table_name_blocked))
            print('Done Rename Table [{}] ==> [{}]'.format(_table_name_changenickname, _new_table_name_changenickname))
            print('Exported as CSV files')
            _ed_time = datetime.now()
            print('Spend Time: ', _ed_time - _st_time)


        