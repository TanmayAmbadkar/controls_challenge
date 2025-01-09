import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from tinyphysics import run_rollout
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

files = np.random.choice(os.listdir("./data"), size = 500, replace = False)
X = []
U = []
Y = []

for file in tqdm(files[:1000]):
    rollout, target_lataccel_history, current_lataccel_history, state_history, action_history = run_rollout("./data/"+file, "zero", "./models/tinyphysics.onnx")
    X.append(np.array([np.array([state_history[i].a_ego, state_history[i].v_ego, state_history[i].roll_lataccel, current_lataccel_history[i-1]]) for i in range(1, len(state_history))]))
    U.append(np.array(action_history[1:]))
    Y.append(np.array(current_lataccel_history[1:]))

for file in tqdm(files[:1000]):
    rollout, target_lataccel_history, current_lataccel_history, state_history, action_history = run_rollout("./data/"+file, "pid", "./models/tinyphysics.onnx")
    X.append(np.array([np.array([state_history[i].a_ego, state_history[i].v_ego, state_history[i].roll_lataccel, current_lataccel_history[i-1]]) for i in range(1, len(state_history))]))
    U.append(np.array(action_history[1:]))
    Y.append(np.array(current_lataccel_history[1:]))

X = np.vstack(X)
U = np.hstack(U).reshape(-1, 1)
Y = np.hstack(Y).reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)


# Convert to PyTorch tensors
X_torch = torch.from_numpy(X_scaled).float()
U_torch = torch.from_numpy(U).float()
Y_torch = torch.from_numpy(Y_scaled).float()

# Concatenate state + action for a 5D input
XU_torch = torch.cat([X_torch, U_torch], dim=1)  # shape (N,5)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(XU_torch, Y_torch)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ---------------------------------------------------------D
# 2. Define a simple neural network surrogate
#    Input  = 5 dims (4 state + 1 action)
#    Output = 1 dim (next lataccel)
# ---------------------------------------------------------
class SurrogateNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu2 = nn.GELU()
        self.fc4 = nn.Linear(hidden_dim, 1)  # single output

    def forward(self, x):
        x = self.gelu1(self.fc1(x))
        x = self.gelu2(self.fc2(x))
        # x = self.gelu3(self.fc3(x))
        x = self.fc4(x)
        return x
    
    

net = SurrogateNet()

# print(net(XU_torch[:1])) # test the network with a single input

# ---------------------------------------------------------
# 3. Train the network
# ---------------------------------------------------------
optimizer = optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.MSELoss()


num_epochs = 100
for epoch in range(num_epochs):
    r2 = []
    losses = []
    for batch_x, batch_y in loader:
        
        y_pred = net(batch_x)
        loss = criterion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        r2.append(r2_score(batch_y.detach().numpy(), y_pred.detach().numpy()))
        losses.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs}, Loss = {np.mean(losses):.5f}")
        print(f"Epoch {epoch+1:2d}/{num_epochs}, r2 score = {np.mean(r2):.5f}")
    

# Now `net` is your trained surrogate model

torch.save(net.state_dict(), 'model_weights.pth')

pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))
pickle.dump(scaler_Y, open("scaler_Y.pkl", "wb"))