#mac setup
install anaconda with python 3.6
install tensorflow
install magenta

pip install magenta




#running base line version
arbitrary_image_stylization_with_weights \
  --checkpoint=/path/to/arbitrary_style_transfer/model.ckpt \
  --output_dir=/path/to/output_dir \
  --style_images_paths=images/style_images/*.jpg \
  --content_images_paths=images/content_images/*.jpg \
  --image_size=256 \
  --content_square_crop=False \
  --style_image_size=256 \
  --style_square_crop=False \
  --logtostderr
  
#running web cam version
#by default this needs a usb webcam plugged in

python3 ./arbitrary_style/arbitrary_image_stylization_cam.py \
 --checkpoint=./arbitrary_style_transfer/model.ckpt \
 --output_dir=output/ \
 --style_images_paths=images/style_images/aha_drawing.jpeg  \
 --content_images_paths=images/content_images/*.jpg  \
 --image_size=256   --content_square_crop=False \
 --style_image_size=256  \
 --style_square_crop=False   --logtostderr

# jetson tx2 configuration notes
1) use the latest Jetpack (4.2.1 at the time of writing this)
2) this uses python3 - so use pip3 for all pip installations
3) install the following using apt-get:
 python3-llvmlite
 python3-numba
 python3-llvmlite
 python3-cffi
 python3-sklearn
 python3-matplotlib
 python3-scipy
 libfreetype6-dev
 emacs
4) pip3 install
   sonnet sk-video sox mir-eval joblib IPython intervaltree bokeh backports.tempfile
   pypng oauth2client kfac gym gunicorn gin-config gevent bz2file
   librosa --no-deps
   tensorflow-probability tensorflow-datasets sympy six scipy requests tqdm mesh-tensorflow  flask
   tensor2tensor --no-deps
   magenta-gpu --no-deps
   dataclasses
   pillow
   django-imagekit
   django_celery_results
   celery
   screeninfo
   numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta
5) install tensorflow from nvidia distro for jetson (google it)
