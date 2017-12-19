from models import *
from utils.tf_utils import set_global_seed, make_session, init_graph
from utils.logger import Logger


class Trainer(object):
    def __init__(self, obs_dim, config):
        self.model = select_ae(config.which_ae)(obs_dim, config)
        self.sess = make_session()
        self.logger = Logger(log_dir="logs/", var_list=self.model._params)
        set_global_seed(config.seed)
        init_graph(self.sess)

    def summarize(self, stats, batch):
        summaries, global_step = self.sess.run([self.model._summaries, self.model.global_step], feed_dict={self.model.x:batch})

        self.logger.dump(stats=stats, tf_summary=summaries, global_step=global_step)

    def save(self, global_step):
        self.logger.save_model(sess=self.sess, global_step=global_step)

    def encode(self, x):
        return self.sess.run(self.model.h_hat, feed_dict={self.model.x: x})

    def decode(self, x):
        return self.sess.run(self.model.x_hat, feed_dict={self.model.x: x})

    def latent(self, x):
        assert self.model.scope == "vae" or self.model.scope == "vqvae"
        return self.sess.run(self.model.z, feed_dict={self.model.x: x})

    def vq(self, x):
        assert self.model.scope == "vqvae"
        return self.sess.run(self.model.e, feed_dict={self.model.x: x})

    def train(self, batch):
        assert self.model.scope != "stacked_ae"
        loss, _, = self.sess.run([self.model.loss, self.model.train_op], feed_dict={self.model.x:batch})
        return loss

    def train_stacked(self, batch):
        assert self.model.scope == "stacked_ae"
        losses = []
        for key in sorted(self.model.train_schedule.keys()):
            loss, = self.sess.run(list(self.model.train_chedule[key].values()), feed_dict={self.model.x:batch})
            losses.append(loss)






