import tensorflow as tf

class Config():
    batch_size = 64
    train_steps = 10000
    log_path = "logs/"
    lr = 1e-3
    clip = 40.
    optim = tf.train.AdamOptimizer
    z_dim = 10
    dict_size =64
    depth = 2
    which_ae = "VAE"
    noise_type = None
    # TODO add CNN
    topology = dict(_type = "linear", _arch= [128,64,32])

    # VQVAE
    beta = 0.25
    # DAE
    drop_prob = .8
    # RAE
    sparsity_level = .5
    act = tf.nn.relu
    # def _act(self):
    #     return tf.nn.relu

