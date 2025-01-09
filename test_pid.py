from tinyphysics import *
from controllers import pid
import os
from sb3_contrib import RecurrentPPO
# from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import constant_fn
from controllers import pid
from tqdm import tqdm
from torch.utils.data.dataset import Dataset, random_split
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sb3_contrib.common.recurrent.type_aliases import RNNStates


datapath = "./data"
files = os.listdir(datapath)

control = pid.Controller()


env = TinyPhysicsSimulatorEnv(files)


rewards = []
for i in tqdm(range(10)):
    done = False
    rew = 0
    obs, _ = env.reset()
    while not done:
        action = control.update(obs[4], obs[0], None, None)
        obs, reward, terminated, truncated, info = env.step([action])
        done = terminated or truncated
        rew += reward
        if done:
            rewards.append(rew)
            break
        
print(np.mean(rewards))