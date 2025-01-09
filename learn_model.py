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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pytorch_lightning as pl

files = np.random.choice(os.listdir("./data"), size = 1000, replace = False)
X = []
U = []
Y = []

def process_file(file_path, controller):
    """
    Helper function that runs the rollout for a single file and
    returns the computed X, U, Y arrays.
    """
    rollout, target_lataccel_history, current_lataccel_history, state_history, action_history = run_rollout(
        file_path, controller, "./models/tinyphysics.onnx"
    )

    # Create the feature array (X_item)
    X_item = np.array([
        [
            state_history[i].a_ego,
            state_history[i].v_ego,
            state_history[i].roll_lataccel,
            current_lataccel_history[i-1],
        ]
        for i in range(1, len(state_history))
    ])

    # Create the action array (U_item)
    U_item = np.array(action_history[1:])

    # Create the target array (Y_item)
    Y_item = np.array(current_lataccel_history[1:])

    return X_item, U_item, Y_item

# Example usage
files_subset = files[:1000]  # or however many you want to process
X, U, Y = [], [], []

with ProcessPoolExecutor(max_workers=32) as executor:  # Adjust max_workers as needed
    # Submit a batch of futures
    futures = {executor.submit(process_file, f"./data/{file}", "zero"): file for file in files_subset}

    # Use as_completed to iterate over completed tasks
    for future in tqdm(as_completed(futures), total=len(futures)):
        file_name = futures[future]
        try:
            x_item, u_item, y_item = future.result()
            X.append(x_item)
            U.append(u_item)
            Y.append(y_item)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


with ProcessPoolExecutor(max_workers=32) as executor:  # Adjust max_workers as needed
    # Submit a batch of futures
    futures = {executor.submit(process_file, f"./data/{file}", "pid"): file for file in files_subset}

    # Use as_completed to iterate over completed tasks
    for future in tqdm(as_completed(futures), total=len(futures)):
        file_name = futures[future]
        try:
            x_item, u_item, y_item = future.result()
            X.append(x_item)
            U.append(u_item)
            Y.append(y_item)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")



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
    
class SurrogateLightningModule(pl.LightningModule):
    def __init__(self, input_dim=5, hidden_dim=128, lr=1e-5):
        super().__init__()
        # This saves hparams (nice for logging/hyperparameter tuning)
        self.save_hyperparameters()

        # Instantiate your original network
        self.model = SurrogateNet(input_dim=input_dim, hidden_dim=hidden_dim)

        # Loss function
        self.criterion = nn.MSELoss()

        # Learning rate
        self.lr = lr

    def forward(self, x):
        """Forward pass simply calls the SurrogateNet."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        - Receives a batch from the train DataLoader
        - Forward pass
        - Compute loss and metric
        - Log them
        - Return the loss (so Lightning can do the backprop)
        """
        x, y = batch  # batch is (features, targets)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar = True)
       
        return loss

    def validation_step(self, batch, batch_idx):
        """
        - Receives a batch from the val DataLoader
        - Compute loss and metric
        - Log them
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Define and return your optimizer(s) and LR scheduler(s).
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


model = SurrogateLightningModule(
    input_dim=5,   # Or whatever matches your data
    hidden_dim=128,
    lr=1e-5
)

# Initialize a trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",       # "cpu", "gpu", or "auto"
    devices=1 if torch.cuda.is_available() else None,
)

# Fit/train
trainer.fit(
    model,
    train_dataloaders=loader,
)

# # print(net(XU_torch[:1])) # test the network with a single input

# # ---------------------------------------------------------
# # 3. Train the network
# # ---------------------------------------------------------
# optimizer = optim.Adam(net.parameters(), lr=0.00001)
# criterion = nn.MSELoss()


# num_epochs = 100
# for epoch in range(num_epochs):
#     r2 = []
#     losses = []
#     for batch_x, batch_y in loader:
        
#         y_pred = net(batch_x)
#         loss = criterion(y_pred, batch_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         r2.append(r2_score(batch_y.detach().numpy(), y_pred.detach().numpy()))
#         losses.append(loss.item())
#     if (epoch+1) % 5 == 0:
#         print(f"Epoch {epoch+1:2d}/{num_epochs}, Loss = {np.mean(losses):.5f}")
#         print(f"Epoch {epoch+1:2d}/{num_epochs}, r2 score = {np.mean(r2):.5f}")
    

# # Now `net` is your trained surrogate model

torch.save(model.model.state_dict(), 'model_weights.pth')

pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))
pickle.dump(scaler_Y, open("scaler_Y.pkl", "wb"))
