# Generated by Django 2.0.5 on 2018-08-04 22:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('style_manager', '0005_auto_20180804_2226'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='style',
            name='model',
        ),
        migrations.AddField(
            model_name='style',
            name='source_file',
            field=models.FileField(default='', upload_to='source_files'),
        ),
    ]
