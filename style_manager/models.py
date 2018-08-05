from django.db import models
from django.contrib.auth import get_user_model
from imagekit.models import ImageSpecField
from imagekit.processors import ResizeToFill


class Style(models.Model):
    # author = models.ForeignKey(
    #     get_user_model(),
    #     on_delete=models.CASCADE
    # )
    title = models.CharField(max_length=20, default="useless title")
    source_file = models.ImageField(upload_to="source_files", default="")
    style_model_name = models.CharField(default="", unique=True, max_length=50)
    thumbnail = ImageSpecField(source='source_file',
                                      processors=[ResizeToFill(100, 100)],
                                      format='JPEG',
                                      options={'quality': 60})
    pub_date = models.DateTimeField('date published', auto_now_add=True)
    is_ready = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)