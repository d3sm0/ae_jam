from trainer import Trainer
from config import Config
import numpy as np


def main():
    obs_dim = [Config.batch_size, 20]
    trainer = Trainer(obs_dim = obs_dim, config = Config)

    for step in range(Config.maxsteps):
        batch = np.random.normal(size = (Config.batch_size, 20))
        loss = trainer.train(batch)
        trainer.summarize(stats =dict(loss = loss, step = step), batch=batch)
        if step % Config.save_freq == 0:
            trainer.save(global_step=step)


if __name__ == '__main__':
    main()