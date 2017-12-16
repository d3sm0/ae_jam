import tensorflow as tf

from util import make_noise, fc, conv, convT, flatten


class Autoencoder(object):
    def __init__(self, obs_dim, config, scope="autoencoder", link=None):
        self.scope = scope
        self.batch_size = config.batch_size
        self._optim = config.optim
        if link is not None:
            self.x = link
        else:
            self._init_ph(obs_dim)
        self.global_step = tf.train.get_or_create_global_step()

    def _init_ph(self, obs_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + [obs_dim])

    def _add_noise(self, drop_prob, noise_type):
        self._drop_prob = tf.placeholder_with_default(tf.constant(drop_prob), shape=[])
        self._is_training = tf.placeholder_with_default(tf.constant(0), shape=[])

        x_tilde = make_noise(x=self.x, drop_prob=self._drop_prob, noise_type=noise_type)
        x = tf.cond(tf.cast(self._is_training, dtype=tf.bool), lambda: self.x, lambda: x_tilde)
        return x

    def _init_graph(self, obs_dim, topology, act, noise_type=None, drop_prob=None):
        with tf.variable_scope(self.scope):
            x = self.x
            if noise_type is not None:
                x = self._add_noise(drop_prob=drop_prob, noise_type=noise_type)
            self.h_hat = self._build_encoder(x=x, topology=topology, act=act)
            self.x_hat = self._build_decoder(enc=self.h_hat, obs_dim=obs_dim, topology=topology,
                                             act=act)

    def _train_op(self, lr):
        self.train_op = self._optim(lr).minimize(self.loss)

    def _loss_op(self):
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat))

    @staticmethod
    def _trainable_variables(scope="autoencoder"):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    @staticmethod
    def _build_encoder(x, topology, act):
        h = tf.reshape(x, [-1,28,28,1]) # TODO bad hack for mnist
        # h= x
        with tf.variable_scope("encoder"):
            if topology["_type"] == "linear":
                for idx, size in enumerate(topology["_arch"]):
                    h = fc(h, scope="h_{}".format(idx), units=size, act=act)
            elif topology["_type"] == "cnn":
                assert h.get_shape().ndims == 4
                # use (-1,1,D,1) for 1D Conv
                for idx, (num_filters, kernel_size, stride) in enumerate(topology["_arch"]):
                    h = conv(x=h, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                             scope="conv_{}".format(idx))
        return h

    @staticmethod
    def _build_decoder(enc, obs_dim, topology, act):
        h_hat = enc
        with tf.variable_scope("decoder"):
            if topology["_type"] == "linear":
                for idx, size in enumerate(reversed(topology["_arch"][1:])):
                    h_hat = fc(h_hat, scope="h_{}".format(idx), units=size, act=act)
            elif topology["_type"] == "cnn":
                # assert h_hat.get_shape().ndims == 4
                if h_hat.get_shape().ndims <4:
                    h_hat = tf.expand_dims(tf.expand_dims(h_hat, 1),1)
                # use (-1,1,D,1) for 1D Conv
                for idx, (num_filters, kernel_size, stride) in enumerate(reversed(topology["_arch"][1:])):
                    h_hat = convT(x=h_hat, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                                  scope="conv_{}".format(idx))
            x_hat = fc(flatten(h_hat), scope="logits", units=obs_dim, act=tf.nn.sigmoid)
        return x_hat


class VAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):  # h_size, n_p=2, act=tf.nn.elu, lr=1e-3):
        super().__init__(obs_dim, config=config, scope="vae", link=link)
        self.z_dim = config.z_dim
        self._init_graph(obs_dim=obs_dim, topology=config.topology, act=tf.nn.relu, drop_prob=config.drop_prob,
                         noise_type=config.noise_type)
        self._loss_op()
        self._train_op(config.lr)

    @staticmethod
    def _build_z(enc, z_dim):
        with tf.variable_scope("latent"):
            z_mu= fc(x=enc, scope="mu_z", units=z_dim, act=lambda x: x)
            z_log_sigma_sq = fc(x=enc, scope="log_sigma_sq_z", units=z_dim, act=lambda x: x)
            eps = tf.random_normal(shape=tf.shape(z_mu), mean=0, stddev=1, dtype=tf.float32)
            z = z_mu + tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
        return z_mu, z_log_sigma_sq, z

    def _init_graph(self, obs_dim, topology, act, noise_type=None, drop_prob=None):
        with tf.variable_scope(self.scope):
            x = self.x
            if noise_type is not None:
                x = self._add_noise(drop_prob=drop_prob, noise_type=noise_type)
            self.h_hat = self._build_encoder(x=x, topology=topology, act=act)
            self.z_mu, self.z_log_sigma_sq, self.z = self._build_z(enc=flatten(self.h_hat), z_dim=self.z_dim)
            self.x_hat = self._build_decoder(enc=self.z, obs_dim=obs_dim, topology=topology, act=act)

    def _loss_op(self):
        eps = 1e-5
        entropy = -tf.reduce_sum(
            self.x * tf.log(eps + self.x_hat) + (1 - self.x) * tf.log(eps + 1 - self.x_hat), axis=-1
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
        self.z_dim = config.z_dim  # would be k dimension in the paper
        self.dict_size = config.dict_size
        self.beta = config.beta
        self._init_graph(obs_dim=obs_dim, topology=config.topology, act=config.act, drop_prob=config.drop_prob,
                         noise_type=config.noise_type)
        self._loss_op()
        self._train_op(config.lr)

    @staticmethod
    def _build_z(enc, z_dim, dict_size):
        with tf.variable_scope("latent"):
            e = tf.get_variable("embeddings", shape=[z_dim, dict_size], initializer=tf.random_normal_initializer(.1))

            # batch_size, latent_h, latent_w, K, D
            # D: dictionary size and K dim of the latent space
            z_e = tf.tile(tf.expand_dims(enc, -2), [1, z_dim, 1])
            # embbedding
            e_exp = tf.reshape(e, [1, z_dim, dict_size])
            k = tf.argmin(tf.norm(z_e - e_exp, axis=-1), axis=-1)  # [latent_h, latent_w, D]
            z_q = tf.gather(e, k)
        return e, z_q

    def _init_graph(self, obs_dim, topology, act, noise_type=None, drop_prob=None):
        with tf.variable_scope(self.scope):
            x = self.x
            if noise_type is not None:
                x = self._add_noise(drop_prob=drop_prob, noise_type=noise_type)
            self.z_e = self._build_encoder(x=x, topology=topology, act=act)
            self.e, self.z_q = self._build_z(enc=self.z_e, z_dim=self.z_dim, dict_size=self.dict_size)
            # x_hat = p_x_z
            self.x_hat = self._build_decoder(enc=self.z_q, obs_dim=obs_dim, topology=topology, act=act)

    def _loss_op(self):
        self.vq_loss = tf.reduce_mean(tf.norm(tf.stop_gradient(self.z_e) - self.z_q, axis=-1) ** 2)
        self.commit_loss = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(self.z_q), axis=-1) ** 2)
        # elf.neg_log- (tf.log(self.x_hat + 1e-5) - tf.log(1/tf.cast(self.k_dim, tf.float32)))
        self.recon_loss = tf.reduce_mean(tf.square(self.x_hat - self.x))
        #
        # should do all dimension [1,2,3]
        # self.recon_loss = - (tf.reduce_mean(tf.log(self.x_hat)) - tf.log(tf.cast(self.z_dim, tf.float32)))
        self.loss = self.recon_loss + self.vq_loss + self.beta * self.commit_loss

    def _train_op(self, lr):
        decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/decoder")
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/encoder")

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

        self.train_op = self._optim(learning_rate=lr).apply_gradients(
            dec_gvs + enc_gvs + embed_gvs, global_step=self.global_step
        )


class AE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):
        super().__init__(obs_dim, config=config, scope="simple_ae", link=link)
        self._init_graph(obs_dim, topology=config.topology, act=config.act, drop_prob=config.drop_prob,
                         noise_type=config.noise_type)
        self._loss_op()
        self._train_op(config.lr)


class RAE(Autoencoder):
    def __init__(self, obs_dim, config, link=None):
        super().__init__(obs_dim, config=config, scope="residual_ae", link=link)
        self.beta = config.beta  # 1e-3
        self.sparsity_level = config.sparsity_level  # .5
        self._init_ph(obs_dim=obs_dim)
        self._init_graph(obs_dim=obs_dim, topology=config.topology, act=config.act, drop_prob=config.drop_prob,
                         noise_type=config.noise_type)
        self._loss_op()
        self._train_op(config.lr)

    def _train_op(self, lr):
        regularizer = self.kl_divergence(self.sparsity_level, self.h_hat)
        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat)) + self.beta * tf.reduce_mean(regularizer)
        self.train_op = self._optim(self._lr).minimize(self.loss)

    @staticmethod
    def kl_divergence(p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)


class DAE(Autoencoder):
    def __init__(self, obs_dim, config,
                 link=None):  # h_size, act=tf.nn.sigmoid, lr=1e-2, noise_type="zeros", drop_prob=1.):
        super().__init__(obs_dim, config=config, scope="denoise_ae", link=link)
        self._init_graph(obs_dim=obs_dim, topology=config.topology, act=config.act, drop_prob=config.drop_prob,
                         noise_type=config.noise_type)
        self._loss_op()
        self._train_op(config.lr)


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
    def __init__(self, obs_dim, config,
                 link):  # , h_size, act=tf.nn.sigmoid, lr=1e-2, drop_prob=1., noise_type="none"):
        super().__init__(obs_dim, config=config, scope="stacked_ae")

        self._build_stack(obs_dim=obs_dim, config=config, link=link)
        self._train_op(config.lr)

    def _build_stack(self, obs_dim, config, link=None):
        which_ae = config.which_ae
        Ae = select_ae(which_ae)  # return a class
        ae_0 = Ae(obs_dim=obs_dim, config=config, link=link)
        self.stack = [(ae_0.loss, ae_0.train_op)]
        h = ae_0.h_hat
        with tf.variable_scope("SAE"):
            for idx in range(config.depth):
                with tf.variable_scope("{}_{}".format(which_ae, idx)):
                    ae = Ae(obs_dim=h.get_shape()[1], config=config, link=tf.stop_gradient(h))
                    h = ae.h_hat
                    self.stack.append((ae.loss, ae.train_op))

    def _train_op(self, lr):
        self.train_schedule = {}
        for idx, (loss, train_op) in enumerate(self.stack):
            self.train_schedule["ae_{}".format(idx)] = {"loss": loss, "train_op": train_op}
