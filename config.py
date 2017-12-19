import tensorflow as tf

class Config():

    # Network params
    topology = dict(_type = "linear", _arch= [128,64,32])

    # Training params
    clip = 40.
    lr = 1e-3
    optim = tf.train.AdamOptimizer
    act = tf.nn.relu
    batch_size = 64
    maxsteps = int(1e5)
    seed = 123

    # Model params
    save_freq = 100
    log_path = "logs/"

    # Ae specific params
    which_ae = "AE"
    # VAE / VQVAE
    z_dim = 10
    dict_size =64
    beta = 0.25
    # DAE
    noise_type = None
    drop_prob = .8
    # RAE
    sparsity_level = .5
    # SAE
    depth = 2
