# region imports
import os
import random
import numpy as np
import torch
# endregion


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cuDNN methods are non-deterministic even after seeding unless explicitly set to deterministic mode.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_name(config):
    pure_env_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
    return f"{config.Models.WorldModel.Backbone}_{config.Models.Agent.Policy}_{pure_env_name}_seed{config.BasicSettings.Seed}"


class _NoOpRun():
    """Mimics the wandb run handle (id/name/dir) when logging is disabled."""

    def __init__(self, run_name):
        self.id = "no-wandb"
        self.name = run_name
        self.dir = os.getcwd()


class WandbLogger():
    """Logs training metrics to Weights & Biases."""

    def __init__(self, config, project=None, mode='online'):
        import wandb
        self.run = wandb.init(project=project, config=config, mode=mode, name=_run_name(config))
        self.run.name = f"{self.run.name}_{self.run.id}"
        self.tag_step = {}

    def log(self, tag, value, global_step):
        import wandb
        if "video" in tag:
            wandb.log({tag: wandb.Video(value, fps=1, format='gif')}, step=global_step)
        elif "images" in tag:
            images = [wandb.Image(img) for img in value]
            wandb.log({tag: images}, step=global_step)
        elif "hist" in tag:
            wandb.log({tag: wandb.Histogram(value)}, step=global_step)
        else:
            wandb.log({tag: value}, step=global_step)

    def update_config(self, update_dict):
        import wandb
        wandb.config.update(update_dict)

    def close(self):
        import wandb
        wandb.finish()


class NoOpLogger():
    """Drop-in logger used when wandb is disabled; same interface, no side effects."""

    def __init__(self, config, project=None, mode='online'):
        self.run = _NoOpRun(_run_name(config))
        self.tag_step = {}

    def log(self, tag, value, global_step):
        return

    def update_config(self, update_dict):
        return

    def close(self):
        return


def make_logger(config, project=None, mode='online'):
    if config.BasicSettings.get('UseWandb', True):
        return WandbLogger(config, project=project, mode=mode)
    return NoOpLogger(config, project=project, mode=mode)


class EMAScalar():

    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar
