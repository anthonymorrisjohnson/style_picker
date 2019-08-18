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