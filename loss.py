import tensorflow as tf
import os
import numpy as np
import inspect

def hinge_gan_loss(discriminator_data, discriminator_z):
    loss_discriminator_data = tf.reduce_mean(tf.nn.relu(1 - discriminator_data))
    loss_discriminator_z = tf.reduce_mean(tf.nn.relu(1 + discriminator_z))
    loss_discriminator = (loss_discriminator_data + loss_discriminator_z)

    loss_generator_adversarial = -tf.reduce_mean(discriminator_z)
    return loss_discriminator, loss_generator_adversarial
def adversarial_loss(outputs, is_real, is_disc=None, type='nsgan'):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    outputs = tf.reshape(outputs, [-1])
    if type == 'hinge':
        if is_disc:
            if is_real:
                outputs = -outputs
            return tf.reduce_mean(tf.nn.relu(1 + outputs))
        else:
            return tf.reduce_mean(-outputs)

    elif type == 'nsgan':
        labels = tf.ones_like(outputs) if is_real else tf.zeros_like(outputs)
        loss = tf.keras.metrics.binary_crossentropy(labels, outputs)
        return loss
    elif type == 'lsgan':
        labels = tf.ones_like(outputs) if is_real else tf.zeros_like(outputs)
        loss = tf.keras.metrics.mean_squared_error(labels, outputs)
        return loss

def l1_loss(inputs, targets):
    inputs = tf.reshape(inputs, [-1])
    targets = tf.reshape(targets, [-1])
    loss = tf.reduce_mean(tf.abs(inputs - targets))
    return loss

def tv_loss_mask(y_comp, mask, margin=3):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""

    # Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = tf.ones([margin, margin, tf.shape(mask)[3], tf.shape(mask)[3]])
    dilated_mask = tf.nn.conv2d(mask, kernel, strides=[1,1,1,1], padding='SAME')

    # Cast values to be [0., 1.], and compute dilated hole region of y_comp
    dilated_mask = tf.cast(dilated_mask > 0, tf.float32)
    # tf.debugging.assert_less(tf.reduce_sum(mask, axis=[1,2,3]), tf.reduce_sum(dilated_mask, axis=[1,2,3]))
    P = dilated_mask * y_comp

    # Calculate total variation loss
    a = l1_loss_mask(P[:,1:,:,:], P[:,:-1,:,:], dilated_mask)
    b = l1_loss_mask(P[:,:,1:,:], P[:,:,:-1,:], dilated_mask)
    return a + b

def tv_loss(inputs):
    r""" A smooth loss in fact.
    Like the smooth prior in MRF. V(y) = || y_{n+1} - y_n ||_1
    """
    dy = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dx = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    dy_loss = l1_loss(dy, tf.zeros_like(dy))
    dx_loss = l1_loss(dx, tf.zeros_like(dx))
    return dy_loss + dx_loss

def l1_loss_mask(inputs, targets, mask):
    loss = tf.reduce_sum(tf.abs(inputs - targets), axis=[1,2,3])
    xs = targets.get_shape().as_list()
    # xs = tf.cast(xs, tf.float32)
    ratio = tf.reduce_sum(mask, axis=[1,2,3]) * xs[3]  # mask: BWH1. if mask = BWH3, then remove '*3'
    loss_mean = tf.reduce_mean(loss / (ratio + 1e-12))  # avoid mask = 0
    return loss_mean

def style_loss(x, y):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    def compute_gram(x):

        b, h, w, ch = x.get_shape().as_list()
        f = tf.reshape(x, [-1, ch, w * h])
        f_T = tf.transpose(f, perm=[0, 2, 1])
        # G = tf.matmul(f, f_T) / (h * w * ch)
        for i in range(b):
            G = tf.matmul(f[i], f_T[i])
            G = tf.expand_dims(G, axis=0)
            if i == 0:
                g = G
            else:
                g = tf.concat([g, G], axis=0)
            g = g / (h * w * ch)
        return g / (h * w * ch)

    x_vgg = Vgg19(x)
    y_vgg = Vgg19(y)

    # Compute loss
    style_loss = 0.0
    style_loss += l1_loss(compute_gram(x_vgg.conv2_2), compute_gram(y_vgg.conv2_2))
    style_loss += l1_loss(compute_gram(x_vgg.conv3_4), compute_gram(y_vgg.conv3_4))
    style_loss += l1_loss(compute_gram(x_vgg.conv4_4), compute_gram(y_vgg.conv4_4))
    style_loss += l1_loss(compute_gram(x_vgg.conv5_2), compute_gram(y_vgg.conv5_2))

    return style_loss

def perceptual_loss(x, y, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    x_vgg = Vgg19(x)
    y_vgg = Vgg19(y)

    content_loss = 0.0
    content_loss += weights[0] * l1_loss(x_vgg.conv1_1, y_vgg.conv1_1)
    content_loss += weights[1] * l1_loss(x_vgg.conv2_1, y_vgg.conv2_1)
    content_loss += weights[2] * l1_loss(x_vgg.conv3_1, y_vgg.conv3_1)
    content_loss += weights[3] * l1_loss(x_vgg.conv4_1, y_vgg.conv4_1)
    content_loss += weights[4] * l1_loss(x_vgg.conv5_1, y_vgg.conv5_1)


    return content_loss


class Vgg19:
    def __init__(self, img, vgg19_npy_path=None):
        with tf.variable_scope('VGG19'):
            if vgg19_npy_path is None:
                path = inspect.getfile(Vgg19)
                path = os.path.abspath(os.path.join(path, os.pardir))
                path = os.path.join(path, "vgg19.npy")
                vgg19_npy_path = path
                # print(vgg19_npy_path)

            self.data_dict = np.load(vgg19_npy_path, encoding='latin1',allow_pickle=True).item()
            # print("npy file loaded")

            self.build(img)

        # def __setitem__(self, key, value):

    def build(self, x):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.conv1_1 = self.conv_layer(x, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")

        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")

        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")

        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")

        self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
