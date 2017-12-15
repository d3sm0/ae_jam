import tensorflow as tf

import numpy as np

from misc import make_noise

class Config(object):
    batch_size = 64
    train_steps = 10000
    log_path = "logs/"
    lr = 1e-2
    clip = 40.
    optim = tf.train.AdamOptimizer
    act = tf.nn.relu
    z_dim = None
    stack_depth = 0
    noise_type = None
    # TODO add CNN
    topology = {"type": ["linear", [64, 128]], "cnn": ["lol"]}
    # VQVAE
    beta = 0.25
    # DAE
    drop_prob = .8
    # RAE
    sparsity_level = .5


class Autoencoder(object):
    def __init__(self, obs_dim, config, scope="autoencoder", link=None):
        self.scope = scope
        self.batch_size = config.batch_size
        self._optim = config.optim
        self._act = config.act
        self._lr = config.lr
        self._noise_type = config.noise_type
        self._drop_prob = config.drop_prob
        if link is not None:
            self.x = link
        else:
            self._init_ph(obs_dim)
        self.global_step = tf.train.get_or_create_global_step()

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + list(obs_dim))
        if self._noise_type is not None:
            self._drop_prob = tf.placeholder_with_default(tf.constant(self._drop_prob), shape=[])
            self._is_training = tf.placeholder_with_default(tf.constant(0), shape=[])

    def _init_graph(self, obs_dim, topology, act):
        with tf.variable_scope(self.scope):
            x = self.x
            if self._noise_type is not None:
                x_tilde = make_noise(x=x, drop_prob=self._drop_prob, noise_type=self._noise_type)
                x = tf.cond(tf.cast(self._is_training, dtype=tf.bool), lambda: self.x, lambda: x_tilde)
                if topology["type"] == "linear":
                    self.h_hat = self._build_encoder(x=x, topology=topology["linear"][1], act=act)
                    self.x_hat = self._build_decoder(enc=self.x_hat, obs_dim=None, topology=topology["linear"][1],
                                                     act=act)
                else:
                    raise NotImplementedError()

    def _train_op(self):
        self.train_op = self._optim(self._lr).minimize(self.loss)

    def _loss_op(self):
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat))

    @staticmethod
    def _trainable_variables(scope="autoencoder"):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    @staticmethod
    def _build_encoder(x, topology, act):

        h = x
        with tf.variable_scope("encoder"):
            for idx, size in enumerate(topology):
                h_hat = tf.layers.dense(inputs=h, units=topology, activation=act,
                                        kernel_initializer=tf.random_normal_initializer(stddev=.1),
                                        name="enc_h_{}".format(idx))
        return h_hat

    @staticmethod
    def _build_decoder(enc, obs_dim, topology, act):
        h_hat = enc
        with tf.variable_scope("decoder"):
            for idx, size in enumerate(reversed(topology[1:])):
                h_hat = tf.layers.dense(inputs=h_hat, units=size, activation=act,
                                        kernel_initializer=tf.random_normal_initializer(stddev=.1),
                                        name="dec_h".format(idx))

            x_hat = tf.layers.dense(inputs=h_hat, units=obs_dim, activation=tf.nn.sigmoid,
                                    kernel_initializer=tf.random_normal_initializer(stddev=.1), name="logits")
        return x_hat


class VAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):  # h_size, n_p=2, act=tf.nn.elu, lr=1e-3):
        super().__init__(obs_dim, config=config, scope="vae", link=link)
        self.z_dim = config.z_dim
        self._eps = config.eps
        self._init_graph(obs_dim, topology=config.topology, act=config.act)
        self._loss_op()
        self._train_op()

    @staticmethod
    def _build_z(enc, z_dim):
        with tf.variable_scope("latent"):
            z_mu = tf.layers.dense(inputs=enc, units=z_dim, activation=lambda x: x)
            z_log_sigma_sq = tf.layers.dense(inputs=enc, units=z_dim, activation=lambda x: x)
            eps = tf.random_normal(shape=tf.shape(z_mu), mean=0, stddev=1, dtype=tf.float32)
            z = z_mu + tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
        return z_mu, z_log_sigma_sq, z

    def _init_graph(self, obs_dim, topology, act):
        with tf.variable_scope(self.scope):
            self.h_hat = self._build_encoder(x=self.x, topology=topology, act=act)
            self.z_mu, self.z_log_sigma_sq, self.z = self._build_z(enc=self.x_hat, z_dim=self.z_dim)
            self.x_hat = self._build_decoder(enc=self.z, obs_dim=obs_dim, topology=topology, act=act)

    def _loss_op(self):
        entropy = -tf.reduce_sum(
            self.x * tf.log(self._eps + self.x_hat) + (1 - self.x) * tf.log(self._eps + 1 - self.x_hat), axis=-1
        )
        self.regularizer = tf.reduce_mean(entropy)
        # p is the noise distribution and z is N(0,1)
        kl_p_z = - 0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1)
        self.recon_loss = tf.reduce_mean(kl_p_z)
        self.loss = self.regularizer + self.recon_loss


class VQVAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):  # , h_size, e_dims=(64, 5), act=tf.nn.relu, lr=1e-2, beta=.25):
        super().__init__(obs_dim, config=config, scope="VQVAE", link=link)
        self.dict_size, self.z_dim = config.e_dims  # would be k dimension in the paper
        self.beta = config.beta
        self._init_ph(obs_dim=obs_dim)
        self._init_graph(obs_dim, topology=config.topology, act=config.act)
        self._loss_op()
        self._train_op()

    @staticmethod
    def _build_z(enc, z_dim, dict_size):
        with tf.variable_scope("latent"):
            e = tf.get_variable("embeddings", shape=[z_dim, dict_size], initializer=tf.random_normal_initializer(.1))

            # batch_size, latent_h, latent_w, K, D
            # D: dictionary size and K dim of the latent space
            z_e = tf.tile(tf.expand_dims(enc, -2), [1, z_dim, 1])
            # embbedding
            e = tf.reshape(e, [1, z_dim, dict_size])
            k = tf.argmin(tf.norm(z_e - e, axis=-1), axis=-1)  # [latent_h, latent_w, D]
            z_q = tf.gather(e, k)
        return e, z_q

    def _init_graph(self, obs_dim, topology, act):
        with tf.variable_scope(self.scope):
                self.z_e = self._build_encoder(x=self.x, topology=topology, act=act)
                self.e, self.z_q = self._build_z(enc=self.z_e, z_dim=self.z_dim, dict_size=self.dict_size)
                # x_hat = p_x_z
                self.x_hat = self._build_decoder(enc=self.z_q, obs_dim=obs_dim, topology=topology, act=act)
    def _loss_op(self):

        self.vq_loss = tf.reduce_mean(tf.stop_gradient(self.z_e) - self.z_q) ** 2
        self.commit_loss = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(self.z_q)) ** 2)
        # elf.neg_log- (tf.log(self.x_hat + 1e-5) - tf.log(1/tf.cast(self.k_dim, tf.float32)))
        self.recon_loss = tf.reduce_mean(tf.square(self.x_hat - self.x))
        #
        # should do all dimension [1,2,3]
        self.recon_loss = - (tf.reduce_mean(tf.log(self.x_hat)) - tf.log(tf.cast(self.z_dim, tf.float32)))
        self.loss = self.recon_loss + self.vq_loss + self.beta * self.commit_loss

    def _train_op(self):
        decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/encoder")
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/decoder")

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

        enc_gvs = list(zip(enc_grads, encoder_params))

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(
            dec_gvs + enc_gvs + embed_gvs, global_step=self.global_step
        )


class AE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):
        super().__init__(obs_dim, config=config, scope="simple_ae", link=link)
        self._init_ph(obs_dim=obs_dim)
        self._init_graph(obs_dim, topology=config.topology, act=config.act)
        self._loss_op()
        self._train_op()


class RAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):
        super().__init__(obs_dim, config=config, scope="residual_ae", link=link)
        self.beta = config.beta  # 1e-3
        self.sparsity_level = config.sparsity_level  # .5
        self._init_ph(obs_dim=obs_dim)
        self._init_graph(obs_dim, topology=config.topology, act=config.act)
        self._loss_op()
        self._train_op()

    def _train_op(self):
        regularizer = self.kl_divergence(self.sparsity_level, self.h_hat)
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat)) + self.beta * tf.reduce_mean(regularizer)
        self.train_op = self._optim(self._lr).minimize(self.loss)

    @staticmethod
    def kl_divergence(p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)


class DAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):  # h_size, act=tf.nn.sigmoid, lr=1e-2, noise_type="zeros", drop_prob=1.):
        super().__init__(obs_dim, config=config, scope="denoise_ae", link=link)
        self._noise_type = config.noise_type
        self._drop_prob = config.drop_prob
        self._init_ph(obs_dim=obs_dim)
        self._init_graph(obs_dim, topology=config.topology, act=config.act)
        self._loss_op()
        self._train_op()

    def _init_graph(self, obs_dim, topology, act):
        x_tilde = make_noise(x=self.x, drop_prob=self._drop_prob, noise_type=self._noise_type)
        x = tf.cond(tf.cast(self._is_training, dtype=tf.bool), lambda: self.x, lambda: x_tilde)
        self.x_hat, self.h_hat = self._build_encoder(x=x, topology=topology, act=act)


def select_ae(ae):
    if ae == "AE":
        return AE
    elif ae == "VAE":
        return VAE
    elif ae == "VQVAE":
        return VQVAE
    elif ae == "DAE":
        return DAE
    elif ae == "RAE":
        return RAE
    else:
        raise NotImplementedError()


class SAE(Autoencoder):
    def __init__(self, obs_dim, config):  # , h_size, act=tf.nn.sigmoid, lr=1e-2, drop_prob=1., noise_type="none"):
        super().__init__(obs_dim, config=config, scope="stacked_ae", link=None)

        self._build_stack(obs_dim=obs_dim, config=config)
        self._train_op()

    def _build_stack(self, obs_dim, config):
        which_ae = config.which_ae
        Ae = select_ae(which_ae)  # return a class

        self.stack = [Ae(obs_dim=obs_dim, config=config)]
        h = self.stack[0].h_hat
        with tf.variable_scope("SAE"):
            for idx in range(config.n_ae):
                with tf.variable_scope("{}_{}".format(which_ae, idx)):
                    ae = Ae(obs_dim=h.get_shape()[1], config=config, link=tf.stop_gradient(h))
                    h = ae.h_hat
                    self.stack.append((ae.loss, ae.train_op))

    def _train_op(self):
        self.train_schedule = {}
        for idx, (x_hat, loss, train_op) in enumerate(self.stack):
            self.train_schedule["ae_{}".format(idx)] = {"loss": loss, "train_op": train_op}

# TODO fix trainer
class Trainer(object):
    def __init__(self, enc_type, network_params):
        if enc_type == "AE":
            self.model = AE(None, )
        elif enc_type == "RAE":
            self.model = RAE(None, )
        elif enc_type == "DAE":
            self.model = DAE(None, )
        elif enc_type == "SAE":
            self.model = SAE(None, )
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


# TODO design tests for each class
