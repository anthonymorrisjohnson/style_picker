from __future__ import absolute_import, unicode_literals
from celery import task
import subprocess

@task()
def process_style(file_path):
    print("processing in the background" + file_path)
    subprocess.check_output(['ls', '-l'])
    # run deep learning model
    # wait a long time
    # upload to s3?