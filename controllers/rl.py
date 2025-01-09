from . import BaseController
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import cvxpy as cp
import pickle


class SurrogateNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu2 = nn.GELU()
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.gelu3 = nn.GELU()
        self.fc4 = nn.Linear(hidden_dim, 1)  # single output

    def forward(self, x):
        x = self.gelu1(self.fc1(x))
        x = self.gelu2(self.fc2(x))
        # x = self.gelu3(self.fc3(x))
        x = self.fc4(x)
        return x



class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self):
        
        self.model = PPO.load("ppo_tinyphysics1")
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        net = SurrogateNet()
        net.load_state_dict(torch.load('model_weights.pth',  weights_only=True))
        self.prediction_model = net
        self.scale_X = pickle.load(open("scaler_X.pkl", "rb"))
        self.scale_y = pickle.load(open("scaler_y.pkl", "rb"))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        
        
        curr_state = np.zeros(10)
        curr_state[0] = current_lataccel
        curr_state[1] = state.v_ego
        curr_state[2] = state.a_ego
        curr_state[3] = state.roll_lataccel
        if len(future_plan.lataccel) > 4:
            curr_state[4:] = np.concatenate([[target_lataccel],future_plan.lataccel[:5]])
        elif len(future_plan.lataccel) > 0:
            curr_state[4:] = np.concatenate([[target_lataccel],future_plan.lataccel, future_plan.lataccel[-1]*np.ones(5-len(future_plan.lataccel))])
        else:
            curr_state[4:] = np.concatenate([[target_lataccel], target_lataccel*np.ones(5)])
        
        
        action, lstm_states  = self.model.predict(curr_state, state=self.lstm_states,  episode_start=self.episode_starts, deterministic=True)
        self.lstm_states = lstm_states
        self.episode_starts = np.zeros((1,), dtype=bool)
        
        c, A, B = self.finite_difference_jacobian(self.prediction_model, self.scale_X.transform(np.array([[state.a_ego, state.v_ego, state.roll_lataccel, current_lataccel]]))[0], action)
        
        N = 1
        x0 = self.scale_X.transform(np.array([[state.a_ego, state.v_ego, state.roll_lataccel, current_lataccel]]))[0][3]
        target_latacc = self.scale_y.transform(np.array([[target_lataccel]]))[0, 0]
        v_ego_seq = self.scale_X.transform(np.array([[state.a_ego, state.v_ego, state.roll_lataccel, current_lataccel]]))[0][1]
        a_ego_seq = self.scale_X.transform(np.array([[state.a_ego, state.v_ego, state.roll_lataccel, current_lataccel]]))[0][0]
        roll_seq = self.scale_X.transform(np.array([[state.a_ego, state.v_ego, state.roll_lataccel, current_lataccel]]))[0][2]
        c_array = c
        A_array = A
        B_array = B
        u_RL = action[0]
        latacc_weight = 5
        jerk_weight = 0
        rl_weight = 1
        u_min = -2
        u_max = 2
        if len(future_plan.lataccel) > 0:
            future_latacc = self.scale_y.transform(np.array([[future_plan.lataccel[0]]]))[0, 0]
        else:
            future_latacc = -1000
        action_mpc = self.solve_latacc_mpc_qp(N, x0, target_latacc, future_latacc, v_ego_seq, a_ego_seq, roll_seq, c_array, A_array, B_array, latacc_weight, jerk_weight, rl_weight, u_RL, u_min, u_max)
        if action_mpc is None:
            action_mpc = u_RL[0]
        print(action_mpc)
        return action_mpc
    
    def local_linear_approx(self, net, x_bar, u_bar):
        """
        Compute a local linear model for net(x,u) around the point (x_bar, u_bar).
        
        Parameters
        ----------
        net    : SurrogateNet (PyTorch model)
        x_bar  : numpy array of shape (4,) for the state
        u_bar  : float or shape (1,) for the action
        
        Returns
        -------
        c : float
            The constant term in the linear approximation
        A : numpy array of shape (4,)
            The partial derivative w.r.t. x
        B : float
            The partial derivative w.r.t. u
        """
        # 1. Create a single input tensor [x_bar, u_bar] of shape (1,5)
        x_bar_torch = torch.tensor(x_bar, dtype=torch.float32)
        u_bar_torch = torch.tensor(u_bar, dtype=torch.float32).view(-1)  # shape (1,) if scalar
        xu_bar = torch.cat([x_bar_torch, u_bar_torch], dim=0).view(1, -1)  # shape (1,5)
        
        # 2. Mark it as requiring gradient
        xu_bar.requires_grad_(True)
        
        # 3. Forward pass
        y_bar = net(xu_bar)  # shape (1,1)
        
        # 4. Backprop to get partial derivatives
        y_bar.backward()  # fills xu_bar.grad with partials d(y_bar)/d(xu_bar)
        
        grads = xu_bar.grad.data.numpy().flatten()  # shape (5,)
        
        # grads[0..3] is partial wrt x_bar, grads[4] is partial wrt u_bar
        A = grads[0:4]  # partial derivative w.r.t. x
        B = grads[4]    # partial derivative w.r.t. u
        
        # 5. c = f(x_bar,u_bar) - A*x_bar - B*u_bar
        f_val = y_bar.detach().item()  # scalar
        c = f_val - (A @ x_bar) - (B * u_bar)
        
        return c, A, B
    
    def finite_difference_jacobian(self, net, x_bar, u_bar, delta=1e-4):
        """
        Approximate partial derivatives wrt the 5D input [x_bar, u_bar].
        """
        import numpy as np
        
        # We'll define a helper function that returns net([x,u]) as a scalar
        def f(xu):
            xu_tensor = torch.tensor(xu, dtype=torch.float32).unsqueeze(0)
            y = net(xu_tensor)
            return y.item()
        
        # Combine x_bar + u_bar into a single vector
        base_input = np.concatenate([x_bar, u_bar], axis=0)  # shape (5,)
        
        # Evaluate function at base
        f_base = f(base_input)
        
        grad = np.zeros_like(base_input)
        for i in range(len(base_input)):
            # +delta
            x_plus = base_input.copy()
            x_plus[i] += delta
            f_plus = f(x_plus)
            
            # -delta
            x_minus = base_input.copy()
            x_minus[i] -= delta
            f_minus = f(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2*delta)
        
        # Now grad[0..3] is partial wrt x_bar, grad[4] wrt u_bar
        A = grad[0:4]
        B = grad[4]
        
        c = f_base - A @ x_bar - B * u_bar
        return c, A, B

    def solve_latacc_mpc_qp(
        self,
        N,
        x0,
        target_latacc,
        future_latacc,
        v_ego_seq,
        a_ego_seq,
        roll_seq,
        # local linearization results
        c_array,    # shape (N,)
        A_array,    # shape (N,4)
        B_array,    # shape (N,)
        # weighting
        latacc_weight = 1.0,
        jerk_weight = 10.0,
        rl_weight = 2.0,
        u_RL = 0.05,
        # optional steering bounds
        u_min=None, 
        u_max=None
    ):
        """
        Solve a horizon-based QP:

        min sum_t [ latacc_weight*(x[t] - target_latacc[t])^2
                    + jerk_weight*(x[t] - x[t-1])^2 ]
            + rl_weight*(u[0] - u_RL)^2

        subject to
        x[0] = x0
        x[t+1] = c_t + A_t*[vEgo, aEgo, roll, x[t]] + B_t*u[t]
                (with vEgo, aEgo, roll known from v_ego_seq, a_ego_seq, roll_seq)
        optional:  u_min <= u[t] <= u_max
        """
        # 1) Define Variables
        u = cp.Variable()    # u[0..N-1]

        constraints = []
        # 2) x[0] = x0


        # c_t + A_t*[vEgo, aEgo, roll, x[t]] + B_t*u[t]
        # We parse out A_t's portion for each input
        #   A_t[0] -> partial wrt vEgo
        #   A_t[1] -> partial wrt aEgo
        #   A_t[2] -> partial wrt roll
        #   A_t[3] -> partial wrt x[t] (current latacc)
        rhs = (c_array
            + A_array[0]*v_ego_seq
            + A_array[1]*a_ego_seq
            + A_array[2]*roll_seq
            + A_array[3]*x0
            + B_array*u)

        # 4) If steering bounds
        if u_min is not None:
            constraints.append(u >= u_min)
        if u_max is not None:
            constraints.append(u <= u_max)

        # 5) Define Cost
        cost_terms = []
        cost_terms.append(latacc_weight * cp.square(rhs - target_latacc))
        # b) jerk (discrete derivative of x)
        cost_terms.append(jerk_weight * cp.square(rhs - x0))
        # if future_latacc != -1000:
        #     cost_terms.append(jerk_weight * cp.square(rhs - future_latacc))
        # c) closeness to RL's first action
        cost_terms.append(rl_weight * cp.square(u - u_RL))

        total_cost = cp.sum(cost_terms)

        # 6) Solve QP
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Solver did not converge. Status:", problem.status)
            return None

        return u.value

  
