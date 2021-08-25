import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
# import tensorflow.contrib as tf_contrib
# from tensorflow.contrib.framework.python.ops import add_arg_scope
# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

# weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
# weight_regularizer = None

##################################################################################
# Layer
##################################################################################
def images_summary(images, name, max_outs, color_format='BGR'):
    """Summary images.

    **Note** that images should be scaled to [-1, 1] for 'RGB' or 'BGR',
    [0, 1] for 'GREY'.

    :param images: images tensor (in NHWC format)
    :param name: name of images summary
    :param max_outs: max_outputs for images summary
    :param color_format: 'BGR', 'RGB' or 'GREY'
    :return: None
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        if color_format == 'BGR':
            img = tf.clip_by_value(
                (tf.reverse(images, [-1])+1.)*127.5, 0., 255.)
        elif color_format == 'RGB':
            img = tf.clip_by_value((images+1.)*127.5, 0, 255)
        elif color_format == 'GREY':
            img = tf.clip_by_value(images*255., 0, 255)
        else:
            raise NotImplementedError("color format is not supported.")
        tf.summary.image(name, img, max_outputs=max_outs)
@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.leaky_relu ,training = True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x
@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME',training = True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output
    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding)
    return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
    

def sndis_conv(x, cnum, ksize=5, stride=2, padding='SAME', name='conv', training=True):
    """Define conv for sn-patch discriminator.
    Activation is set to leaky_relu.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    with tf.variable_scope(name):
        in_channel = x.get_shape().as_list()[-1]
        kernel = tf.get_variable('kernel', [ksize, ksize, in_channel, cnum],
                                 initializer=tf.variance_scaling_initializer(), trainable=training)
        x = tf.nn.conv2d(x, spectral_norm(kernel), strides=[1, stride, stride, 1],
                padding=padding, name=name)
        x = tf.nn.leaky_relu(x)
    return x


def res_block(x, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'res_block'):
    cnum = x.get_shape().as_list()[-1]
    xin = x
    x = tf.layers.conv2d(x, cnum // 4, kernel_size = 1, strides = 1, activation = activation, padding = padding, name = name + '_conv1')
    x = tf.layers.conv2d(x, cnum // 4, kernel_size = 3, strides = 1, activation = activation, padding = padding, name = name + '_conv2')
    x = tf.layers.conv2d(x, cnum, kernel_size = 1, strides = 1, activation = None, padding = padding, name = name + '_conv3')
    x = tf.add(xin, x, name = name + '_add')
    x = tf.layers.batch_normalization(x, name = name + '_bn')
    x = activation(x, name = name + '_out')
    return x

def hinge_gan_loss(discriminator_data, discriminator_z):
    loss_discriminator_data = tf.reduce_mean(tf.nn.relu(1 - discriminator_data))
    loss_discriminator_z = tf.reduce_mean(tf.nn.relu(1 + discriminator_z))
    loss_discriminator = (loss_discriminator_data + loss_discriminator_z)
    loss_generator_adversarial = -tf.reduce_mean(discriminator_z)
    return loss_discriminator, loss_generator_adversarial

def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x

