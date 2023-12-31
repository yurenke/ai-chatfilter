# Generated by Django 3.0.7 on 2023-02-16 08:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai', '0012_auto_20200903_1412'),
    ]

    operations = [
        migrations.CreateModel(
            name='NicknameTextbook',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('origin', models.CharField(default='', max_length=512)),
                ('text', models.CharField(default='', max_length=255)),
                ('status', models.IntegerField(default=0)),
            ],
        ),
        migrations.AddIndex(
            model_name='nicknametextbook',
            index=models.Index(fields=['status'], name='ai_nickname_status_1a4632_idx'),
        ),
    ]
