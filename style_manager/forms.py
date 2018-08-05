from django import forms
from .models import Style

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = Style
        fields = ('title', 'source_file', 'style_model_name')
