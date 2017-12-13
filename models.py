import tensorflow as tf

import numpy as np


class Autoencoder(object):
    def __init__(self):
        self.global_step = tf.train.get_or_create_global_step()

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])

    def _build_graph(self, h_size, act, scope="autoencoder"):
        with tf.variable_scope(scope):
            self.x_hat, self.h_hat = self._build_ae(x=self.x, h_size=h_size, act=act)

    def _train_op(self, lr):
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat))
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(self.loss)

    @staticmethod
    def _trainable_variables(scope="autoencoder"):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    @staticmethod
    def _loss_op(x, x_hat, lr, reg):
        # TODO break train op? Maybe?
        raise NotImplementedError()

    @staticmethod
    def _build_ae(x, h_size, act):
        with tf.variable_scope("encoder"):
            h_hat = tf.layers.dense(inputs=x, units=h_size, activation=act,
                                    kernel_initializer=tf.random_normal_initializer(stddev=.1), name="enc_h")
        with tf.variable_scope("decoder"):
            # for idx, size in enumerate(reversed(h_size[1:])):
            x_hat = tf.layers.dense(inputs=h_hat, units=x.get_shape()[1], activation=act,
                                    kernel_initializer=tf.random_normal_initializer(stddev=.1), name="dec_h")
        return x_hat, h_hat



class VAE(Autoencoder):
    def __init__(self, obs_dim, h_size, n_p=2, act=tf.nn.elu, lr=1e-3):
        self.n_p = n_p
        self._init_ph(obs_dim=obs_dim)
        self._build_graph(h_size=h_size, act=act)
        self._train_op(lr=lr)
        pass

    def _build_graph(self, h_size, act, scope="vae"):
        x = self.x
        with tf.variable_scope(scope):
            with tf.variable_scope("encoder"):
                h = tf.layers.dense(inputs=x, units=h_size, activation=act)
                self.p_mu = tf.layers.dense(inputs=h, units=self.n_p, activation=None)
                self.p_log_sigma_sq = tf.layers.dense(inputs=h, units=self.n_p, activation=None)
                eps = tf.random_normal(shape=tf.shape(self.p_mu), mean=0, stddev=1, dtype=tf.float32)
                self.p = self.p_mu + tf.multiply(tf.sqrt(tf.exp(self.p_log_sigma_sq)), eps)
            with tf.variable_scope("decoder"):
                self.x_hat = tf.layers.dense(inputs=self.p, units=x.get_shape()[1], activation=tf.nn.sigmoid)

    def _train_op(self, lr):
        eps = 1e-10
        entropy = -tf.reduce_sum(
            self.x * tf.log(eps + self.x_hat) + (1 - self.x) * tf.log(eps + 1 - self.x_hat), axis=-1
        )
        self.entropy = tf.reduce_mean(entropy)
        # p is the noise distribution and z is N(0,1)
        kl_p_z = - 0.5 * tf.reduce_sum(
            1 + self.p_log_sigma_sq - tf.square(self.p_mu) - tf.exp(self.p_log_sigma_sq)
            , axis=-1)
        self.kl_p_z = tf.reduce_mean(kl_p_z)
        self.loss = self.entropy + self.kl_p_z
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)


class AE(Autoencoder):
    def __init__(self, obs_dim, h_size, act=tf.nn.sigmoid, lr=1e-2):
        # super().__init__(self, obs_dim, h_size, act, lr)
        self._init_ph(obs_dim=obs_dim)
        self._build_graph(h_size, act, scope="AE")
        self._train_op(lr=lr)


class RAE(Autoencoder):
    def __init__(self, obs_dim, h_size, act=tf.nn.sigmoid, lr=1e-5, sparsity_level=.5, beta=1e-3):
        self.beta = beta
        self.sparsity_level = sparsity_level
        self._init_ph(obs_dim=obs_dim)
        self._build_graph(h_size, act, scope="RAE")
        self._train_op(lr=lr)

    def _train_op(self, lr):
        regularizer = self.kl_divergence(self.sparsity_level, self.h_hat)
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat)) + self.beta * tf.reduce_mean(regularizer)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(self.loss)

    @staticmethod
    def kl_divergence(p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)


class DAE(Autoencoder):
    def __init__(self, obs_dim, h_size, act=tf.nn.sigmoid, lr=1e-2, noise_type="zeros", drop_prob=1.):
        self._noise_type = noise_type
        self._drop_prob = drop_prob
        self._init_ph(obs_dim=obs_dim)
        self._build_graph(h_size, act, scope="DAE")
        self._train_op(lr)

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
        self._drop_prob = tf.placeholder_with_default(tf.constant(self._drop_prob), shape=[])
        self._is_training = tf.placeholder_with_default(tf.constant(0), shape=[])

    def _build_graph(self, h_size, act, scope="autoencoder"):
        with tf.variable_scope(scope):
            x_tilde = make_noise(x=self.x, drop_prob=self._drop_prob, noise_type=self._noise_type)
            x = tf.cond(tf.cast(self._is_training, dtype=tf.bool), lambda: self.x, lambda: x_tilde)
            self.x_hat, self.h_hat = self._build_ae(x=x, h_size=h_size, act=act)


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


class SAE(Autoencoder):
    def __init__(self, obs_dim, h_size, act=tf.nn.sigmoid, lr=1e-2, drop_prob=1., noise_type="none"):
        self._noise_type = noise_type
        self._drop_prob = drop_prob
        self._init_ph(obs_dim=obs_dim)
        self._build_stack(h_size=h_size, act=act)
        self._train_op(lr=lr)

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
        self._drop_prob = tf.placeholder_with_default(tf.constant(self._drop_prob), shape=[])
        self._is_training = tf.placeholder_with_default(tf.constant(0), shape=[])

    def _build_stack(self, h_size, act):
        self.stack = []
        h = self.x
        with tf.variable_scope("SAE"):
            for idx, size in enumerate(h_size):
                # TODO test with denoising
                # x_corrupted = make_noise(x=h, drop_prob=self._drop_prob, noise_type=self._noise_type)
                # h = tf.cond(tf.cast(self._is_training, dtype=tf.bool), lambda: h, lambda: x_corrupted)
                with tf.variable_scope("ae_{}".format(idx)):
                    x_tilde, h = self._build_ae(tf.stop_gradient(h), size, act)
                    self.stack.append((x_tilde, h))

    def _train_op(self, lr):
        self.train_schedule = {}
        x = self.x
        for idx, s in enumerate(self.stack):
            x_tilde, h = s
            loss = tf.reduce_mean(tf.square(x - x_tilde))
            train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
            self.train_schedule["ae_{}".format(idx)] = {"loss": loss, "train_op": train_op}

            x = h


class Trainer(object):
    def __init__(self, enc_type, network_params):
        if enc_type == "AE":
            self.model = AE(**network_params)
        elif enc_type == "RAE":
            self.model = RAE(**network_params)
        elif enc_type == "DAE":
            self.model = DAE(**network_params)
        elif enc_type == "SAE":
            self.model = SAE(**network_params)
        else:
            raise NotImplementedError()
        self.enc_type = enc_type
        self.saver = tf.train.Saver(self.model._trainable_variables(self.enc_type))
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
            for m in mask:
                x_tilde[idx][m] = self._corrupt((x_min, x_max))
        return x_tilde

    def _salty(self, min_max):
        x_min, x_max = min_max
        if np.random.random() > .5:
            return x_max
        else:
            return x_min

class VQVAE(object):
    def __init__(self, obs_dim, h_size, e_dims=(64, 5), act=tf.nn.relu, lr=1e-2, beta=.25):
        self.dict_size, self.k_dim = e_dims
        self.beta = beta
        self.global_step = tf.train.get_or_create_global_step()
        self._init_ph(obs_dim= obs_dim)
        self._build_graph(h_size=h_size, act=act, scope="vqvae")
        self._train_op(lr=lr)

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])

    def _build_graph(self, h_size, act, scope="autoencoder"):
        x = self.x
        with tf.variable_scope(scope):
            with tf.variable_scope("embeddings"):
                self.e = tf.get_variable("embeddings", shape=[self.k_dim, self.dict_size],
                                         initializer=tf.random_normal_initializer(.1))
            with tf.variable_scope("encoder"):
                self.z_e = tf.layers.dense(inputs=x, units=h_size, activation=act)
                # batch_size, latent_h, latent_w, K, D
                # D: dictionary size and K dim of the latent space
                z_e = tf.tile(tf.expand_dims(self.z_e, -2), [1,  self.k_dim, 1])
                # embbedding
                e = tf.reshape(self.e, [1 , self.k_dim, self.dict_size])
                k = tf.argmin(tf.norm(z_e - e, axis=-1), axis=-1)  # [latent_h, latent_w, D]
                self.z_q = tf.gather(self.e, k)
            with tf.variable_scope("decoder"):
                # p_x_z
                self.p_x_z = tf.layers.dense(inputs=self.z_q, units=x.get_shape()[1], activation=tf.nn.sigmoid)
        pass

    def _train_op(self, lr):
        self.vq_loss = tf.reduce_mean(tf.stop_gradient(self.z_e) - self.z_q) ** 2
        self.commit_loss = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(self.z_q)) ** 2)
        # elf.neg_log- (tf.log(self.x_hat + 1e-5) - tf.log(1/tf.cast(self.k_dim, tf.float32)))
        self.recon_loss = tf.reduce_mean(tf.square(self.p_x_z- self.x))
        #
        # should do all dimension [1,2,3]
        self.recon_loss = - (tf.reduce_mean(tf.log(self.p_x_z)) - tf.log(tf.cast(self.k_dim, tf.float32)))
        self.loss = self.recon_loss + self.vq_loss + self.beta * self.commit_loss

        decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae/decoder")
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vqvae/encoder")

        dec_grads = tf.gradients(self.recon_loss, decoder_params)
        dec_gvs = list(zip(dec_grads, decoder_params))
        embed_grads = tf.gradients(self.vq_loss, self.e)
        embed_gvs = list(zip(embed_grads, [self.e]))

        grad_z = tf.gradients(self.recon_loss, self.z_q)
        enc_grads = []
        # this can be written using add_n
        for param in encoder_params:
            g = tf.gradients(self.z_e, param, grad_z)[0] + tf.gradients(self.commit_loss, param)[0]
            enc_grads.append(g)

        # enc_grads = tf.gradients(self.recon_loss + self.beta * self.commit_loss, encoder_params)
        enc_gvs = list(zip(enc_grads, encoder_params))

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(
            dec_gvs + enc_gvs + embed_gvs, global_step=self.global_step
        )

