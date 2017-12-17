import tensorflow as tf
import numpy as np


def flatten(x):
    units = np.prod([t.value for t in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, units])
    return x


# def glorot_init(f_in, f_out):
#     l = tf.sqrt(6/(f_in + f_out))
#     l = tf.cast(l, dtype=tf.int32)
#     return tf.random_uniform_initializer(minval =-l, maxval=l, dtype=tf.float32)
def fc(x, scope, units, act=tf.nn.relu):
    with tf.variable_scope(scope):
        d = x.get_shape()[1].value
        w = tf.get_variable("w", [d, units], initializer= tf.glorot_uniform_initializer(dtype=tf.float32))# tf.random_normal_initializer(init_scale))
        b = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


# tf default [batch_size, height, width, channel], cuda default NCHW
# use (1, k) and (1,s) for convolution over time
def conv(x, scope, num_filters, kernel_size, stride, padding='SAME', act=tf.nn.relu, init_scale = 1.):
    # k_h, k_w = kernel_size
    # s_h, s_w = stride
    with tf.variable_scope(scope):
        channels = x.get_shape()[3]
        w = tf.get_variable("w", [kernel_size, kernel_size, channels, num_filters],
                            initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding) + b
        h = act(z)
        return h


def convT(x, scope, num_filters, kernel_size, stride, padding='SAME', act=tf.nn.relu, init_scale=1.0):
    # k_h, k_w = kernel_size
    # s_h, s_w = stride
    with tf.variable_scope(scope):
        height, width, channels = x.get_shape().dims[1:]
        w = tf.get_variable("w", [kernel_size, kernel_size, num_filters, channels],
                            initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
        shape = tf.stack([tf.shape(x)[0],height * stride, width * stride, num_filters])
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


DATA_DIR = 'datasets/cifar10'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
import os


def maybe_download_and_extract():
    import sys, tarfile
    from six.moves import urllib
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    record_bytes = 1 + 32 * 32 * 3

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [1]), tf.int32)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [1],
                         [1 + 32 * 32 * 3]),
        [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def get_image(train=True, num_epochs=None):
    maybe_download_and_extract()
    if train:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]
    else:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    read_input = read_cifar10(filename_queue)
    return tf.cast(read_input.uint8image, tf.float32) / 255.0, tf.reshape(read_input.label, [])
