# Generated by Django 2.2.6 on 2020-01-15 07:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai', '0005_auto_20200115_1403'),
    ]

    operations = [
        migrations.AddField(
            model_name='digitalvocabulary',
            name='pinyin',
            field=models.CharField(default='', max_length=65),
        ),
    ]
