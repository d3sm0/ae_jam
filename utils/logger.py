import csv
import os
from datetime import datetime

from utils.tf_utils import create_writer, create_saver, create_summary


class Logger(object):
    def __init__(self, log_dir, var_list=None):
        unique_name = datetime.now().strftime('%H-%M-%S')
        self.log_path = os.path.join(os.getcwd(), log_dir, unique_name, 'stats')
        self.save_path = os.path.join(os.getcwd(), log_dir, unique_name, 'model')
        self.tf_writer = create_writer(path=self.log_path, suffix=unique_name)
        self.tf_saver = create_saver(var_list)
        self.ep_summary = create_summary()
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        self.write_header = True

    def create_writer(self, stats_list):
        self.file = open(os.path.join(self.log_path, 'stats.csv'), 'w+t')
        self.writer = csv.DictWriter(self.file, fieldnames=['step'] + stats_list)
        self.writer.writeheader()
        self.write_header = False

    def log(self, stats, total_ep):
        print("===== Ep {} ====".format(total_ep))
        # TODO: Fix me in one line with tqdm progress
        for k, v in stats.items():
            try:
                self.ep_summary.value.add(tag=k, simple_value=v)
                print("{}:{}".format(k, round(v, 3)))
            except TypeError:
                print("{}: {}".format(k, v))
        print('\n')

    def dump(self, stats, global_step=None, tf_summary=None):
        if self.write_header == True:
            self.create_writer(stats_list=list(stats.keys()))
        stats['step'] = datetime.now().strftime('%H:%M:%S:%f')
        self.writer.writerow(stats)
        self.tf_writer.add_summary(summary=self.ep_summary, global_step=global_step)
        if tf_summary is not None:
            self.tf_writer.add_summary(tf_summary, global_step=global_step)
        self.tf_writer.flush()
        self.file.flush()

    def load_model(self, sess, ckpt_dir):
        ckpt = self.tf_saver.recover_last_checkpoints(checkpoint_paths=ckpt_dir)
        if ckpt is not None:
            try:
                self.tf_saver.restore(sess=sess, save_path=ckpt)
            except Exception as e:
                print("Can not restore", e)
                raise e

    def save_model(self, sess, global_step=None):
        try:
            self.tf_saver.save(
                sess, save_path=self.save_path + '/model.ckpt', global_step=global_step,
                write_meta_graph=False
            )
        except Exception as e:
            print('Can not save', e)
            raise e
