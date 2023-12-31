# Generated by Django 2.2.6 on 2020-05-07 03:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0002_analyzingdata'),
    ]

    operations = [
        migrations.CreateModel(
            name='UnknownWord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('unknown', models.CharField(max_length=64)),
                ('text', models.CharField(blank=True, max_length=64, null=True)),
                ('status', models.IntegerField(default=1)),
                ('date', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AddIndex(
            model_name='unknownword',
            index=models.Index(fields=['-date'], name='service_unk_date_2de1ea_idx'),
        ),
        migrations.AddIndex(
            model_name='unknownword',
            index=models.Index(fields=['status'], name='service_unk_status_0654cd_idx'),
        ),
    ]
