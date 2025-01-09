# import pandas as pd
# import os
# import numpy as np

# datapath = "./data"
# files = os.listdir(datapath)
# curr_min = 10000*np.ones(6)
# curr_max = -10000*np.ones(6)
# for file in files[:5000]:

#     # pd.read_csv(f"data/{file}").describe().loc[['min', 'max']]
#     curr_min = np.minimum(curr_min, pd.read_csv(f"data/{file}").describe().loc['min'])
#     curr_max = np.maximum(curr_max, pd.read_csv(f"data/{file}").describe().loc['max'])
    
# print(curr_min)
# print(curr_max)

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
# model.learn(5000)
print(model.policy)

# vec_env = model.get_env()
# mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
# print(mean_reward)

# model.save("ppo_recurrent")
# del model # remove to demonstrate saving and loading

# model = RecurrentPPO.load("ppo_recurrent")

# obs = vec_env.reset()
# # cell and hidden state of the LSTM
# lstm_states = None
# num_envs = 1
# # Episode start signals are used to reset the lstm states
# episode_starts = np.ones((num_envs,), dtype=bool)
# while True:
#     action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     episode_starts = dones
#     # vec_env.render("human")