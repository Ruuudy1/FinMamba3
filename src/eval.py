import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os

from utils import seed_np_torch, WandbLogger
import env_wrapper
import agents
from sub_models.world_models import WorldModel
import yaml
from utils import WandbLogger
import pandas as pd

def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(config,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, logger: WandbLogger, global_step=None):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize, num_envs=config.Evaluate.NumEnvs)
    # print("Evaluating Env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(config.Evaluate.NumEnvs)
    current_obs, _ = vec_env.reset()
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)

    atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
    atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
    game_benchmark_df = atari_benchmark_df.get(atari_pure_name)

    episode_idx = 0
    score_table = {"episode": [], "evaluate/score": [], "evaluate/normalised_score": []}
    for algorithm in game_benchmark_df.index[2:]:
        score_table[f"evaluate/normalised_{algorithm}_score"] = []
    with tqdm(total=config.Evaluate.EpisodeNum, desc="Evaluating episodes") as episode_pbar:
        while True:
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                    # action = np.array([action], dtype=int)
                    # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).to(world_model.device)
                    # current_obs_tensor = rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255
                    if world_model.model == 'Transformer':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent[:,-1:], model_context_action[:,-1:], inference_params)
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=True
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)

            obs, reward, done, truncated, info = vec_env.step(action)
            # cv2.imshow("current_obs", process_visualize(obs[0]))
            # cv2.waitKey(10)
            # update current_obs, current_info and sum_reward
            sum_reward += reward
            current_obs = obs

            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                for i in range(config.Evaluate.NumEnvs):
                    if done_flag[i]:
                        episode_score = sum_reward[i]
                        normalised_score = (episode_score - game_benchmark_df['Random']) / (game_benchmark_df['Human'] - game_benchmark_df['Random'])
                        
                        score_table["episode"].append(episode_idx)
                        score_table["evaluate/score"].append(episode_score)
                        score_table["evaluate/normalised_score"].append(normalised_score)

                        for algorithm in game_benchmark_df.index[2:]:
                            denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                            # Check if the denominator is zero
                            if denominator != 0:
                                normalised_score = (sum_reward[i] - game_benchmark_df['Random']) / denominator
                                score_table[f"evaluate/normalised_{algorithm}_score"].append(normalised_score)
                            else:
                                score_table[f"evaluate/normalised_{algorithm}_score"].append(None)

                        sum_reward[i] = 0
                        episode_idx += 1
                        episode_pbar.update(1)  # Update the episode progress bar
                        if episode_idx == config.Evaluate.EpisodeNum:
                            # print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(score_table['evaluate/score'])}" + colorama.Style.RESET_ALL)
                            for key, value in score_table.items():
                                if key != 'episode' and not np.array(value).any() == None:
                                    logger.log(key, np.mean(value), global_step=global_step)
                            return score_table




if __name__ == "__main__":
    from train import parse_args_and_update_config, DotDict, build_world_model, build_agent
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Read the YAML configuration file
    with open('config_files/configure.yaml', 'r') as file:
        config = yaml.safe_load(file)
   
    
    # Parse the arguments and update the configuration
    config = parse_args_and_update_config(config)   

    config = DotDict(config)
    
    # parse arguments
    # print(colorama.Fore.RED + str(config) + colorama.Style.RESET_ALL)

    device = torch.device(config.BasicSettings.Device)

    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    # getting action_dim with dummy env
    dummy_env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    agent = build_agent(config, action_dim, device=device)
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = logger.run.dir

    if config.BasicSettings.SavePath != 'None':
        print('Loading models')
        world_model.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/world_model.pth"))
        agent.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/agent.pth"))
    
    scores_table = eval_episodes(
        config, world_model=world_model, agent=agent, logger=logger)
    