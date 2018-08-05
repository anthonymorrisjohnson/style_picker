from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from .models import Style
from django.shortcuts import render, redirect
from .tasks import process_style
from django.core import serializers
from django.http import JsonResponse


def index(request):
    models = Style.objects.all()

    context = {'models': models}
    return render(request, 'index.html', context)

def activate(request, id):
    #deactivate
    current_model = Style.objects.filter(is_active=True).update(is_active=False)
    new_model = Style.objects.get(pk=id)
    new_model.is_active = True
    new_model.save()

    return redirect('index')

def current_model(request):
    current_model = Style.objects.filter(is_active=True)
    #throw an exception if we don't have anything active
    data = serializers.serialize('json', current_model)
    return JsonResponse(data, safe=False)

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            print("form is valid")
            form.save()
            print('form saved')
            #print(form)
            process_style.delay(form.instance.source_file.url)
            print("submitted to queue")
            return redirect("index")
        else:
            print("wtf")
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})