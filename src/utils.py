import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
import wandb


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])



class WandbLogger():
    def __init__(self, config, project=None, mode='online'):
        """
        Initialize the Logger class.

        Args:
            path (str): Path to the directory where logs will be saved. This can be used to define the run name in W&B.
            project (str, optional): Name of the W&B project. Defaults to None.
        """
        # Initialize a W&B run with the given project and path as the run name
        pure_env_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        run_name = f"{config.Models.WorldModel.Backbone}_{config.Models.Agent.Policy}_{pure_env_name}_seed{config.BasicSettings.Seed}"
        self.run = wandb.init(project=project, config=config, mode=mode, name=run_name)
        self.run.name = f"{self.run.name}_{self.run.id}"
        self.tag_step = {}


    def log(self, tag, value, global_step):
        """
        Log data to Weights & Biases.

        Args:
            tag (str): The tag or label for the data being logged.
            value: The data to be logged. It can be a scalar, image, histogram, or video.
        """
        # Log data based on the type
        if "video" in tag:
            # Log video
            wandb.log({tag: wandb.Video(value, fps=1, format='gif')}, step=global_step)
        elif "images" in tag:
            # Log images
            images = [wandb.Image(img) for img in value]  # Convert each image to a wandb.Image
            wandb.log({tag: images}, step=global_step)
        elif "hist" in tag:
            # Log histogram
            wandb.log({tag: wandb.Histogram(value)}, step=global_step)
        else:
            # Log scalar value
            wandb.log({tag: value}, step=global_step)

    def update_config(self, update_dict):
        """
        Update the configuration with the given parameters.

        Args:
            update_dict (dict): A dictionary containing scalar parameter information to update in the configuration.
        """
        # Update the configuration using wandb.config.update
        wandb.config.update(update_dict)

    def close(self):
        """
        Finalize and close the W&B run.
        """
        # Finish the run
        wandb.finish()



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
