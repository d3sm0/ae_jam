from models import *


# TODO fix trainer
class Trainer(object):
    def __init__(self, config):
        self.model = select_ae(config.which_ae)
        self.saver = tf.train.Saver(self.model._trainable_variables(self.model.scope))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, dataset, max_steps, batch_size):

        with self.sess.as_default():
            if self.enc_type != "SAE":
                return self._train(x=self.model.x, train_op=self.model.train_op, loss=self.model.loss, dataset=dataset,
                                   max_steps=max_steps)
            else:
                return self._train_stack(dataset=dataset, max_steps=max_steps, batch_size=32)

    def encode(self, x):
        if self.enc_type != SAE:
            h_hat = self.sess.run(self.model.h_hat, feed_dict={self.model.x: x})
            return h_hat
        else:
            # Return only the latest encoding
            x_tilde, h_hat = self.sess.run(self.model.stack[-1], feed_dict={self.model.x: x})
            return h_hat

    @staticmethod
    def _train(x, train_op, loss, dataset=None, max_steps=10000, batch_size=32):
        # TODO implement general batch training or the Dataset from tf
        # dataset = mnist.train
        sess = tf.get_default_session()
        display_step = int(max_steps / 100)
        losses = []
        for step in range(max_steps):
            batch_x, _ = dataset.next_batch(batch_size)
            _, loss = sess.run([train_op, loss], feed_dict={x: batch_x})
            losses.append(loss)
            if step % display_step == 0:
                print("Step {}: Minibatch Loss: {:.2f}".format(step, loss))

        return np.mean(losses)

    def _train_stack(self, dataset, max_steps, batch_size):
        assert self.enc_type == "SAE"
        dataset = mnist.train
        for key in sorted(self.model.train_schedule.keys()):
            ops = self.model.train_schedule[key]
            l = self._train(x=self.model.x, train_op=ops["train_op"], loss=ops["loss"], dataset=dataset)
            print("Stack {}: Minibatch Avg Loss: {:.2f}".format(key, l))

    def test(self, batch):
        x_tilde = self.sess.run(self.model.x_hat, feed_dict={self.model.x: batch})
        return x_tilde

    def save(self):
        self.saver.save(sess=self.sess, save_path="logs/model.ckpt")


from config import Config

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("datasets/mnist", one_hot=False)
# create dataset
features_ph = tf.placeholder(mnist.train.images.dtype, mnist.train.images.shape)
labels_ph = tf.placeholder(mnist.train.labels.dtype, mnist.train.labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((features_ph, labels_ph))
# crate batches
batch_size = 64
num_epochs = 20


dataset = dataset.repeat(num_epochs)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size=batch_size)
# create an in iterator to iterate over the dataset
iterator = dataset.make_initializable_iterator()

x, y = iterator.get_next()
Config.topology = dict(_type = "cnn", _arch = [(16,4,2), (32, 4,2), (128,4,2)])
ae = AE(obs_dim=x.get_shape()[1].value, link=x, config=Config)
with tf.Session() as sess:
    losses = []
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
        sess.run(iterator.initializer, feed_dict={features_ph: mnist.train.images, labels_ph: mnist.train.labels})
        try:
            l, _ = sess.run((ae.loss, ae.train_op))
            losses.append(l)
            print(l)
        except tf.errors.OutOfRangeError:
            print("ops")
import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()
