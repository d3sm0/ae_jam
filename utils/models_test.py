from models import *
from config import Config
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

Config.topology = dict(_type="cnn", _arch=[(16, 4, 2), (32, 4, 2), (128, 4, 2)])
Config.batch_size = 64
Config.dict_size = 128
Config.noise_type = "zeros"
Config.drop_prob = .4
Config.num_epochs = 1000

mnist = input_data.read_data_sets("datasets/mnist", one_hot=False)
# create dataset
features_ph = tf.placeholder(mnist.train.images.dtype, mnist.train.images.shape)
labels_ph = tf.placeholder(mnist.train.labels.dtype, mnist.train.labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((features_ph, labels_ph))
# crate batches



dataset = dataset.repeat(Config.num_epochs)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size=Config.batch_size)
# def reshape_mnist(x,y):
#     return tf.reshape(x, [-1,28,28,1]), y
# dataset = dataset.map(reshape_mnist)
# create an in iterator to iterate over the dataset
iterator = dataset.make_initializable_iterator()

x, y = iterator.get_next()
ae = VQVAE(obs_dim=[64,28,28,1], link=x, config=Config)
with tf.Session() as sess:
    losses = []
    sess.run(tf.global_variables_initializer())
    for ep in range(Config.num_epochs):
        sess.run(iterator.initializer, feed_dict={features_ph: mnist.train.images, labels_ph: mnist.train.labels})
        try:
            recon_loss, _ = sess.run((ae.loss, ae.train_op), feed_dict={ae._is_training: True})
            losses.append(recon_loss)
        except tf.errors.OutOfRangeError:
            print("ep {} completed with loss {}".format(ep, np.mean(losses)))
            break



    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(ae.x_hat, feed_dict={ae.x: batch_x})  # model._is_training: False})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

print("Losses")
plt.plot(losses)
plt.show()

