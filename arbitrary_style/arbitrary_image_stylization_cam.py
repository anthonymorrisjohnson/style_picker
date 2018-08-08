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

import numpy as np
import tensorflow as tf

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import cv2

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_string('checkpoint', None, 'Path to the model checkpoint.')
flags.DEFINE_string('style_images_paths', None, 'Paths to the style images'
                                                'for evaluation.')
# flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
#                    'for evaluation.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('image_size', 768, 'Image size.')
flags.DEFINE_boolean('content_square_crop', False, 'Wheather to center crop'
                                                   'the content image to be a square or not.')
flags.DEFINE_integer('style_image_size', 768, 'Style image size.')
flags.DEFINE_boolean('style_square_crop', False, 'Wheather to center crop'
                                                 'the style image to be a square or not.')
# flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
#                     'styles to evaluate.')
flags.DEFINE_string('interpolation_weights', '[1.0]', 'List of weights'
                                                      'for interpolation between the parameters of the identity'
                                                      'transform and the style parameters of the style image. The'
                                                      'larger the weight is the strength of stylization is more.'
                                                      'Weight of 1.0 means the normal style transfer and weight'
                                                      'of 0.0 means identity transform.')

FLAGS = flags.FLAGS

def display_np_image(image):
  """Saves an image to disk.

  Args:
    image: 3-D numpy array of shape [image_size, image_size, 3] and dtype
        float32, with values in [0, 1].
    output_file: str, output file.
    save_format: format for saving image (eg. jpeg).
  """
  image = np.uint8(image * 255.0)
  #buf = io.BytesIO()
  image = np.squeeze(image, 0)
  #scipy.misc.imsave(buf, np.squeeze(image, 0), format=save_format)
  #cv2.imshow('frame', image)
  cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
  cv2.moveWindow("window", -2000, -3000)            
  cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  cv2.imshow("window", image)
  #buf.seek(0)
  #f = tf.gfile.GFile(output_file, 'w')
  #f.write(buf.getvalue())
  #f.close()

def main(unused_argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MkDir(FLAGS.output_dir)

    # Instantiate video capture object.
    cap = cv2.VideoCapture(1)

    # Set resolution
    # if resolution is not None:
    x_length, y_length = (768, 768)
    cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
    cap.set(4, y_length)
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))
    print('Resolution is: {0} by {1}'.format(x_new, y_new))

    with tf.Graph().as_default(), tf.Session() as sess:
        # Defines place holder for the style image.
        style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if FLAGS.style_square_crop:
            style_img_preprocessed = image_utils.center_crop_resize_image(
                style_img_ph, FLAGS.style_image_size)
        else:
            style_img_preprocessed = image_utils.resize_image(style_img_ph,
                                                              FLAGS.style_image_size)

        # Defines place holder for the content image.
        content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
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

        if tf.gfile.IsDirectory(FLAGS.checkpoint):
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        else:
            checkpoint = FLAGS.checkpoint
            tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

        init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                                 slim.get_variables_to_restore())
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

        # Gets the list of the input style images.
        style_img_list = tf.gfile.Glob(FLAGS.style_images_paths)
        #if len(style_img_list) > FLAGS.maximum_styles_to_evaluate:
        #    np.random.seed(1234)
        #    style_img_list = np.random.permutation(style_img_list)
        #    style_img_list = style_img_list[:FLAGS.maximum_styles_to_evaluate]

        # Gets list of input content images.
        #content_img_list = tf.gfile.Glob(FLAGS.content_images_paths)

        # if style_i % 10 == 0:
        #tf.logging.info('Stylizing  %s with (%d) %s' %
        #                        ( content_img_name, style_i,
        #                         style_img_name))

        #for style_i, style_img_path in enumerate(style_img_list):
        #if style_i > FLAGS.maximum_styles_to_evaluate:
        #    break
        style_img_path = "../images/style_images/bricks_sq.jpg"
        style_img_name = "bricks"
        style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :
                                                                           3]

        # Saves preprocessed style image.
        style_img_croped_resized_np = sess.run(
        style_img_preprocessed, feed_dict={
                        style_img_ph: style_image_np
                    })
        image_utils.save_np_image(style_img_croped_resized_np,
                                          os.path.join(FLAGS.output_dir,
                                                       '%s.jpg' % (style_img_name)))

        # Computes bottleneck features of the style prediction network for the
        # given style image.
        style_params = sess.run(
                    bottleneck_feat, feed_dict={style_img_ph: style_image_np})

        interpolation_weights = ast.literal_eval(FLAGS.interpolation_weights)

        while True:
            # for content_i, content_img_path in enumerate(content_img_list):
            ret, frame = cap.read()
            content_img_np = frame
            #content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :
            #                                                                        3]

            #content_img_name = os.path.basename(content_img_path)[:-4]
            content_img_name = "webcam"

            # Saves preprocessed content image.
            inp_img_croped_resized_np = sess.run(
                content_img_preprocessed, feed_dict={
                    content_img_ph: content_img_np
                })
            #image_utils.save_np_image(inp_img_croped_resized_np,
            #                          os.path.join(FLAGS.output_dir,
            #                                       '%s.jpg' % (content_img_name)))

            # Computes bottleneck features of the style prediction network for the
            # identity transform.
            identity_params = sess.run(
              bottleneck_feat, feed_dict={style_img_ph: content_img_np})
            

            # Interpolates between the parameters of the identity transform and
            # style parameters of the given style image.
            for interp_i, wi in enumerate(interpolation_weights):
              stylized_image_res = sess.run(
                stylized_images,
                feed_dict={
                  bottleneck_feat:
                  identity_params * (1 - wi) + style_params * wi,
                  content_img_ph:
                  content_img_np
                })
            
            # Saves stylized image.
            #image_utils.save_np_image(
            #  stylized_image_res,
            #  os.path.join(FLAGS.output_dir, '%s_stylized_%s_%d.jpg' %
            #               (content_img_name, style_img_name, interp_i)))
            display_np_image(stylized_image_res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
            #img_out = np.squeeze(stylized_image_res).astype(np.uint8)
            #img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            #cv2.imshow('frame', img_out)
              
    cap.release()
    cv2.destroyAllWindows()


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
