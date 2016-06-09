import skimage.io
import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import numpy as np
from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('original', 'minsk.jpg', 'Original Image')
flags.DEFINE_string('styled', 'tmp_950_color.jpg', 'Styled Image')

original = tf.placeholder("float", [1, 338, 600, 3])
styled = tf.placeholder("float", [1, 338, 600, 3])

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.mul(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.mul(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp


styled_grayscale = tf.image.rgb_to_grayscale(styled)
styled_grayscale_rgb = tf.image.grayscale_to_rgb(styled_grayscale)
styled_grayscale_yuv = rgb2yuv(styled_grayscale_rgb)

original_yuv = rgb2yuv(original)

combined_yuv = tf.concat(3, [tf.split(3, 3, styled_grayscale_yuv)[0], tf.split(3, 3, original_yuv)[1], tf.split(3, 3, original_yuv)[2]])
combined_rbg = yuv2rgb(combined_yuv)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    original_image = skimage.io.imread(FLAGS.original) / 255.0
    original_image = original_image.reshape((1, 338, 600, 3))
    styled_image = skimage.io.imread(FLAGS.styled) / 255.0
    styled_image = styled_image.reshape((1, 338, 600, 3))

    combined_rbg_ = sess.run(combined_rbg, feed_dict={original: original_image, styled: styled_image})

    summary_image = concat_images(original_image.reshape((338, 600, 3)), styled_image.reshape((338, 600, 3)))
    summary_image = concat_images(summary_image, combined_rbg_[0])
    plt.imsave("results.jpg", summary_image)