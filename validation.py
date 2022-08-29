import copy
import glob
import os
import json
import time
from collections import deque

import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_test_args
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from place_env import place_envs
import warnings
warnings.filterwarnings('ignore')


def main():
    args = get_test_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    with open(f'{args.netlist_dir}/info.json') as fp:
        d = json.load(fp)
    num_cell = d['num_cell'] if args.num_cell < 0 else args.num_cell
    num_steps = max(num_cell * 5, args.num_steps)
    envs = place_envs(args.netlist_dir, num_cell, args.grid_size)
    actor_critic = torch.load("./trained_models/placement_300.pt")[0]
    actor_critic.to(device)

    rollouts = RolloutStorage(num_steps, args.num_processes,
                              envs.obs_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(envs.transform(obs))
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)

    features = torch.zeros(num_cell, 2)

    for step in tqdm.tqdm(range(num_steps), total=num_steps):
        # Sample actions
        n = len(envs.results)
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step], features, n)

        # Obser reward and next obs
        obs, done, reward = envs.step(action)
        features[n][0] = action // 32
        features[n][1] = action % 32

        if done:
            obs = envs.reset()
            features = torch.zeros(num_cell, 2)
            print(reward)


if __name__ == "__main__":
    main()
