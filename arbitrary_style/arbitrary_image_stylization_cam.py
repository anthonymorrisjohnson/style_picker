# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates stylized images with different strengths of a stylization.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import sys
import django
from screeninfo import get_monitors
from PIL import ImageFont, ImageDraw, Image

import numpy as np
import tensorflow as tf

sys.path.append(".")
print(sys.path)

from timeit import default_timer as timer
_GPU_MEM_FRACTION = 0.30

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "style_picker.settings")
django.setup()

from style_manager.models import Style

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import cv2

slim = tf.contrib.slim
import tensorflow.contrib.tensorrt as trt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwpb2
from tensorflow.python.client import session as csess
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.ops import array_ops as aops
from tensorflow.python.ops import math_ops as mops
from tensorflow.python.ops import nn as nn
from tensorflow.python.ops import nn_ops as nn_ops

flags = tf.flags
flags.DEFINE_string('checkpoint', None, 'Path to the model checkpoint.')
#flags.DEFINE_string('style_images_paths', None, 'Paths to the style images'
#                                                'for evaluation.')
# flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
#                    'for evaluation.')

#flags.DEFINE_string('output_dir', 'output', 'Output directory.')
flags.DEFINE_boolean('tensorrt', False, 'use tensorrt optimization for inference')
flags.DEFINE_integer('image_size', 768, 'Image size.')
flags.DEFINE_boolean('content_square_crop', False, 'Wheather to center crop'
                                                   'the content image to be a square or not.')
flags.DEFINE_integer('style_image_size', 1200, 'Style image size.')
flags.DEFINE_boolean('style_square_crop', False, 'Wheather to center crop'
                                                 'the style image to be a square or not.')
# flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
#                     'styles to evaluate.')
flags.DEFINE_float('interpolation_weight', 1.0, 'weight'
                                                      'for interpolation between the parameters of the identity'
                                                      'transform and the style parameters of the style image. The'
                                                      'larger the weight is the strength of stylization is more.'
                                                      'Weight of 1.0 means the normal style transfer and weight'
                                                      'of 0.0 means identity transform.')
flags.DEFINE_boolean('showFullScreen', False, 'Whether to show full screen')

FLAGS = flags.FLAGS


def display_np_image(image, fullscreen=False):
    """Saves an image to disk.

    Args:
      image: 3-D numpy array of shape [image_size, image_size, 3] and dtype
          float32, with values in [0, 1].
      output_file: str, output file.
      save_format: format for saving image (eg. jpeg).
    """
    image = np.uint8(image * 255.0)
    # buf = io.BytesIO()
    image = np.squeeze(image, 0)
    # scipy.misc.imsave(buf, np.squeeze(image, 0), format=save_format)
    # cv2.imshow('frame', image)
    #font = cv2.FONT_HERSHEY_PLAIN
    font = ImageFont.truetype("./LEVIBRUSH.TTF", 70)
    helpText = "Bus leaves at 4am"
    #cv2.putText(image, helpText, (10,20), font, 30, (255,255,255), 20)
    #img_out = cv2.putText(image, 'bus leaving the playa at 2AM', (50, 50),
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10,10), helpText, font=font, fill='rgb(0,0,0)')
    #cv2.FONT_HERSHEY_SIMPLEX,
    #                       1.0, (255, 0, 0), 3)
    #for m in get_monitors():
    #    print(str(m))
    image = np.array(img_pil)
    
    if fullscreen:
        #hard code screensize
        blank_image = np.zeros((1080,1920,3), np.uint8)
        blank_image[0:image.shape[0], 0:image.shape[1]] = image
        
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("window", 0, 1000)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", blank_image)
    else:
        cv2.imshow('frame', image)
    # buf.seek(0)
    # f = tf.gfile.GFile(output_file, 'w')
    # f.write(buf.getvalue())
    # f.close()


def main(unused_argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    #if not tf.gfile.Exists(FLAGS.output_dir):
    #    tf.gfile.MkDir(FLAGS.output_dir)

    
    if FLAGS.tensorrt:
        gpu_options = None
        print(trt.trt_convert.get_linked_tensorrt_version())
        gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
        sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
    else:
        sessconfig = None
    
    # Instantiate video capture object.
    cap = cv2.VideoCapture(1)


    # Set resolution
    # if resolution is not None:
    x_length, y_length = (1024, 1280)
    cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
    cap.set(4, y_length)
    cap.read()
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))
    print('Resolution is: {0} by {1}'.format(x_new, y_new))

    with tf.Graph().as_default(), tf.Session(config=sessconfig) as sess:

        #TODO - calculate these dimensions dynamically (they can't use None since TensorRT
        # needs precalculated dimensions
        
        # Defines place holder for the style image.
        style_img_ph = tf.placeholder(tf.float32, shape=[200, 1200, 3], name="style_img_ph")
        if FLAGS.style_square_crop:
            style_img_preprocessed = image_utils.center_crop_resize_image(
                style_img_ph, FLAGS.style_image_size)
        else:
            style_img_preprocessed = image_utils.resize_image(style_img_ph,
                                                              FLAGS.style_image_size)

        # Defines place holder for the content image.
        content_img_ph = tf.placeholder(tf.float32, shape=[200, 1200, 3], name="content_img_ph")
        if FLAGS.content_square_crop:
            content_img_preprocessed = image_utils.center_crop_resize_image(
                content_img_ph, FLAGS.image_size)
        else:
            content_img_preprocessed = image_utils.resize_image(
                content_img_ph, FLAGS.image_size)

        # Defines the model.
        stylized_images, _, _, bottleneck_feat = build_model.build_model(
            content_img_preprocessed,
            style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False)

        print(stylized_images)
        print(bottleneck_feat)

    
        if tf.gfile.IsDirectory(FLAGS.checkpoint):
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        else:
            checkpoint = FLAGS.checkpoint
            tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

        init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                                 slim.get_variables_to_restore())
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

        tf.train.write_graph(sess.graph_def, '.', 'model.pbtxt')
        
        if FLAGS.tensorrt:
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                ['transformer/expand/conv3/conv/Sigmoid'] # The output node names are used to select the usefull nodes
        )
            
            
            trt_graph = trt.create_inference_graph(
                input_graph_def=output_graph_def,
                outputs=["transformer/expand/conv3/conv/Sigmoid"],
                max_workspace_size_bytes=5 << 30,
                max_batch_size=1,
                precision_mode="FP16",  # TRT Engine precision "FP32","FP16" or "INT8"
                minimum_segment_size=10)
            
            bottleneck_feat_O, content_img_ph_O, stylized_images_O = importer.import_graph_def(
                graph_def=trt_graph, return_elements=["Conv/BiasAdd", "content_img_ph", "transformer/expand/conv3/conv/Sigmoid"])
            bottleneck_feat_O = bottleneck_feat_O.outputs[0]
            content_img_ph_O = content_img_ph_O.outputs[0]
            stylized_images_O = stylized_images_O.outputs[0]

            print("bottleneck opt:" + str(bottleneck_feat_O))
            print(content_img_ph_O)
            print(stylized_images_O)
    
        # Gets the list of the input style images.
        #style_img_list = tf.gfile.Glob(FLAGS.style_images_paths)
        # if len(style_img_list) > FLAGS.maximum_styles_to_evaluate:
        #    np.random.seed(1234)
        #    style_img_list = np.random.permutation(style_img_list)
        #    style_img_list = style_img_list[:FLAGS.maximum_styles_to_evaluate]

        # Gets list of input co ntent images.
        # content_img_list = tf.gfile.Glob(FLAGS.content_images_paths)

        # if style_i % 10 == 0:
        # tf.logging.info('Stylizing  %s with (%d) %s' %
        #                        ( content_img_name, style_i,
        #                         style_img_name))

        # for style_i, style_img_path in enumerate(style_img_list):
        # if style_i > FLAGS.maximum_styles_to_evaluate:
        #    break
        interpolation_weight = FLAGS.interpolation_weight       
        activate_style = None
        
        while True:
            start = timer()
            #calculating style isn't the major FPS bottleneck
            current_style = Style.objects.filter(is_active=True).first()
            if (activate_style != current_style):
                activate_style = current_style
                style_img_path = activate_style.source_file.path
                print("current image is " + style_img_path)
                style_img_name = "bricks"
                style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]
                style_image_np = cv2.resize(style_image_np, (1200,200))

                # Saves preprocessed style image.
                style_img_croped_resized_np = sess.run(
                    style_img_preprocessed, feed_dict={
                        style_img_ph: style_image_np
                    })
                #image_utils.save_np_image(style_img_croped_resized_np,
                #                          os.path.join(FLAGS.output_dir,
                #                                       '%s.jpg' % (style_img_name)))
            
                # Computes bottleneck features of the style prediction network for the
                # given style image.
                style_params = sess.run(
                    bottleneck_feat, feed_dict={style_img_ph: style_image_np})


            
            # for content_i, content_img_path in enumerate(content_img_list):
            ret, frame = cap.read()
            print("webcam image: " + str(frame.shape))
            #crop to get the weird 1200x200 format
            content_img_np = frame[500:700, 80:1280 ]
            #content_img_np = frame
            print("cropped image:" + str(content_img_np.shape))
            # content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :
            #                                                                        3]

            # content_img_name = os.path.basename(content_img_path)[:-4]
            content_img_name = "webcam"

            # Saves preprocessed content image.
            print("Input image:" + str(content_img_np.shape))
            inp_img_croped_resized_np = sess.run(
                content_img_preprocessed, feed_dict={
                    content_img_ph: content_img_np
                })
            # image_utils.save_np_image(inp_img_croped_resized_np,
            #                          os.path.join(FLAGS.output_dir,
            #                                       '%s.jpg' % (content_img_name)))

            # Computes bottleneck features of the style prediction network for the
            # identity transform.
            identity_params = sess.run(
                bottleneck_feat, feed_dict={style_img_ph: content_img_np})

            # Interpolates between the parameters of the identity transform and
            # style parameters of the given style image.
            wi = interpolation_weight
            style_np = identity_params * (1 - wi) + style_params * wi
            if FLAGS.tensorrt:
                style_np= np.reshape(style_np, (1,100,1,1))

                stylized_image_res = sess.run(
                    stylized_images_O,
                    feed_dict={
                        bottleneck_feat_O:
                        style_np,
                        content_img_ph_O:
                        content_img_np
                    })
            else:
                stylized_image_res = sess.run(
                    stylized_images,
                    feed_dict={
                        bottleneck_feat:
                        style_np,
                        content_img_ph:
                        content_img_np
                    })

                
            end = timer()
            print(end - start)
            print(stylized_image_res.shape)
            # Saves stylized image.
            # image_utils.save_np_image(
            #  stylized_image_res,
            #  os.path.join(FLAGS.output_dir, '%s_stylized_%s_%d.jpg' %
            #               (content_img_name, style_img_name, interp_i)))
            display_np_image(stylized_image_res, FLAGS.showFullScreen)
            print(stylized_image_res.shape)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #  break
            #img_out = np.squeeze(stylized_image_res).astype(np.uint8)
            #img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            #cv2.imshow('frame', img_out)
            
            
            key = cv2.waitKey(10)
            print("Key " + str(key))
            if key == 27:
                break;
            elif key == 192:
                FLAGS.showFullScreen = False
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif (key == 233 or key == 193):
                FLAGS.showFullScreen = True
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == 60: # less
                interpolation_weight -= 0.25
            elif key == 62: # > more
                interpolation_weight += 0.25

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    cap.release()
    cv2.destroyAllWindows()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
