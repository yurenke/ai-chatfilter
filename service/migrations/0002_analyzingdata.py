# Generated by Django 2.2.6 on 2020-01-06 05:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnalyzingData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.PositiveSmallIntegerField()),
                ('month', models.PositiveSmallIntegerField()),
                ('day', models.PositiveSmallIntegerField()),
                ('good_sentence', models.PositiveIntegerField(default=0)),
                ('blocked_sentence', models.PositiveIntegerField(default=0)),
                ('json_blocked_detail', models.CharField(blank=True, max_length=2048)),
                ('json_addition', models.CharField(blank=True, max_length=1024)),
            ],
        ),
    ]
