import tensorflow as tf
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)


def flatten(x):
    units = np.prod([t.value for t in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, units])
    return x

def fc(x, scope, units, act=tf.nn.relu):
    with tf.variable_scope(scope):
        d = x.get_shape()[1].value
        w = tf.get_variable("w", [d, units], initializer=tf.glorot_uniform_initializer(
            dtype=tf.float32))  # tf.random_normal_initializer(init_scale))
        b = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h

# tf default [batch_size, height, width, channel], cuda default NCHW
# use (1, k) and (1,s) for convolution over time
def conv(x, scope, num_filters, kernel_size, stride, padding='SAME', act=tf.nn.relu, init_scale=1.):
    with tf.variable_scope(scope):
        channels = x.get_shape()[3]
        w = tf.get_variable("w", [kernel_size, kernel_size, channels, num_filters],
                            initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding) + b
        h = act(z)
        return h

def convT(x, scope, num_filters, kernel_size, stride, padding='SAME', act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        height, width, channels = x.get_shape().dims[1:]
        w = tf.get_variable("w", [kernel_size, kernel_size, num_filters, channels],
                            initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
        shape = tf.stack([tf.shape(x)[0], height * stride, width * stride, num_filters])
        z = tf.nn.conv2d_transpose(x, w, output_shape=shape, strides=[1, stride, stride, 1],
                                   padding=padding) + b
        h = act(z)
        return h

def make_noise(x, drop_prob, noise_type):
    binary_tensor = tf.floor(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=1) + drop_prob)

    if noise_type == "gaussian":
        corruption = tf.random_normal(shape=tf.shape(x))
        return x + binary_tensor * corruption
    elif noise_type == "zeros":
        return x * binary_tensor
    elif noise_type == "uniform":
        corruption = tf.random_uniform(shape=tf.shape(x))
        return x + binary_tensor * corruption
    elif noise_type == "none":
        return x
    else:
        raise NotImplementedError()

def set_global_seed(seed=1234):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    pass

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def load_model(sess, load_path, var_list=None):
    ckpt = tf.train.load_checkpoint(ckpt_dir_or_file=load_path)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.restore(sess=sess, save_path=ckpt)
    except Exception as e:
        tf.logging.error(e)

def save(sess, save_path, var_list=None):
    os.makedirs(save_path, exist_ok=True)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.save(sess=sess, save_path=os.path.join(save_path, 'model.ckpt'),
                   write_meta_graph=False)
    except Exception as e:
        tf.logging.error(e)

def create_saver(var_list):
    return tf.train.Saver(var_list=var_list, save_relative_paths=True, reshape=True)

def create_writer(path, suffix):
    return tf.summary.FileWriter(logdir=path, flush_secs=360, filename_suffix=suffix)

def create_summary():
    return tf.summary.Summary()

def init_graph(sess):
    sess.run(tf.global_variables_initializer())
    tf.logging.info('Graph initialized')

def make_config(num_cpu, memory_fraction=.25):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False
    )
    tf_config.gpu_options.allow_growth = False
    return tf_config

def make_session(num_cpu=3):
    return tf.Session(config=make_config(num_cpu=num_cpu))

class Corruptor(object):
    def __init__(self, noise_type):
        if noise_type == "mask":
            self._corrupt = lambda x: 0
        elif noise_type == "salty":
            self._corrupt = self._salty
        elif noise_type == "gaussian":
            self._corrupt = lambda x: np.random.randn
        else:
            raise NotImplementedError

    def corrupt(self, x, p):

        x_tilde = x.copy()
        n, d = x.shape
        x_min = np.min(x)
        x_max = np.max(x)
        for idx in range(n):
            mask = np.random.randint(0, d, p, dtype=np.int32)
            for m in range(mask):
                x_tilde[idx][m] = self._corrupt((x_min, x_max))
        return x_tilde

    def _salty(self, min_max):
        x_min, x_max = min_max
        if np.random.random() > .5:
            return x_max
        else:
            return x_min

def summary_op(t_list, img_list = None):
    ops = []
    for t in t_list:
        name = t.name.replace(':', '_')
        if t.get_shape().ndims < 1:
            op = tf.summary.scalar(name=name, tensor=t)
        else:
            op = tf.summary.histogram(name=name, values=t)
        # op = tf.summary.tensor_summary(name=t.name.split(':')[0], tensor=t)
        ops.append(op)
    if img_list is not None:
        for img in img_list:
            op = tf.summary.image(img.name, img)
            ops.append(op)
    return tf.summary.merge(ops)
