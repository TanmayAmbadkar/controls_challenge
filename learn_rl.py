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

controller = pid.Controller()


env = TinyPhysicsSimulatorEnv(files)

# num_interactions = 200000
# if isinstance(env.action_space, gym.spaces.Box):
#     expert_observations = []
#     expert_actions = []
# obs, _ = env.reset()

# obss, acts = [obs], []
# for i in tqdm(range(num_interactions)):
#     action = control.update(obs[4], obs[0], None, None)
#     acts.append(action)
#     obs, reward, terminated, truncated, info = env.step([action])
#     obss.append(obs)
#     done = terminated or truncated
#     if done:
#         obs, _ = env.reset()
#         expert_observations.append(obss[:-1])
#         expert_actions.append(acts)
#         obss, acts = [obs], []

# class ExpertDataSet(Dataset):
#     def __init__(self, expert_observations, expert_actions):
#         self.observations = expert_observations
#         self.actions = expert_actions

#     def __getitem__(self, index):
#         return (th.Tensor(self.observations[index]), th.Tensor(self.actions[index]))

#     def __len__(self):
#         return len(self.observations)
    
# expert_dataset = ExpertDataSet(expert_observations, expert_actions)

# train_size = int(0.8 * len(expert_dataset))

# test_size = len(expert_dataset) - train_size

# train_expert_dataset, test_expert_dataset = random_split(
#     expert_dataset, [train_size, test_size]
# )

# def pretrain_agent(
#     student,
#     batch_size=64,
#     epochs=1000,
#     scheduler_gamma=0.7,
#     learning_rate=1.0,
#     log_interval=100,
#     no_cuda=True,
#     seed=1,
#     test_batch_size=64,
# ):
#     use_cuda = not no_cuda and th.cuda.is_available()
#     th.manual_seed(seed)
#     device = th.device("cuda" if use_cuda else "cpu")
#     kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

#     if isinstance(env.action_space, gym.spaces.Box):
#         criterion = nn.MSELoss()
#     else:
#         criterion = nn.CrossEntropyLoss()

#     # Extract initial policy
#     model = student.policy.to(device)
    

#     def train(model, device, train_expert_dataset, optimizer):
#         model.train()

#         for batch_idx, (data, target) in enumerate(train_expert_dataset):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             single_hidden_state_shape = (1, 1, 256)
#             lstm_states = RNNStates(
#             (
#                 th.zeros(single_hidden_state_shape, device=device),
#                 th.zeros(single_hidden_state_shape, device=device),
#             ),
#             (
#                 th.zeros(single_hidden_state_shape, device=device),
#                 th.zeros(single_hidden_state_shape, device=device),
#             ),
#         )

#             ep_starts = th.ones((1,))
#             loss = 0
#             for i in range(data.shape[0]):
#                 action, _, _, lstm_states= model(data[i].reshape(1, -1), lstm_states=lstm_states, episode_starts=ep_starts)    
#                 action_prediction = action
#                 ep_starts = th.zeros((1,))
                
#                 loss += criterion(action_prediction, target[i])
#             loss /= data.shape[0]
#             loss.backward()
#             optimizer.step()
#             if batch_idx % log_interval == 0:
#                 print(
#                     "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                         epoch,
#                         batch_idx * len(data),
#                         len(train_loader.dataset),
#                         100.0 * batch_idx / len(train_loader),
#                         loss.item(),
#                     )
#                 )

#     def test(model, device, test_loader):
#         model.eval()
#         test_loss = 0
#         with th.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(device), target.to(device)

#                 if isinstance(env.action_space, gym.spaces.Box):
#                     # A2C/PPO policy outputs actions, values, log_prob
#                     # SAC/TD3 policy outputs actions only
#                     if isinstance(student, (A2C, PPO)):
#                         action, _, _, = model(data)
#                     else:
#                         # SAC/TD3:
#                         action = model(data)
#                     action_prediction = action
#                 else:
#                     # Retrieve the logits for A2C/PPO when using discrete actions
#                     dist = model.get_distribution(data)
#                     action_prediction = dist.distribution.logits
#                     target = target.long()

#                 test_loss = criterion(action_prediction, target)
#         test_loss /= len(test_loader.dataset)
#         print(f"Test set: Average loss: {test_loss:.4f}")

#     # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
#     # and testing
#     train_loader = th.utils.data.DataLoader(
#         dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
#     )
#     test_loader = th.utils.data.DataLoader(
#         dataset=test_expert_dataset,
#         batch_size=test_batch_size,
#         shuffle=True,
#         **kwargs,
#     )

#     # Define an Optimizer and a learning rate schedule.
#     optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
#     scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

#     # Now we are finally ready to train the policy model.
#     for epoch in range(1, epochs + 1):
#         train(model, device, train_expert_dataset, optimizer)
#         # test(model, device, test_loader)
#         scheduler.step()

#     # Implant the trained policy network back into the RL student agent
#     student.policy = model

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, n_steps=2048, epochs = 20, batch_size = 64, learning_rate = 0.0001)
# pretrain_agent(
#     model,
#     epochs=30,
#     scheduler_gamma=0.7,
#     learning_rate=0.0001,
#     log_interval=100,
#     no_cuda=True,
#     seed=1,
#     batch_size=64,
#     test_batch_size=1000,
# )

model.learn(total_timesteps=100000)
env.controller_flag = False
# model.learn(total_timesteps=500000, reset_num_timesteps=False)

model.save("ppo_tinyphysics1")