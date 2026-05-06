import random
import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from typing import Dict
import matplotlib.pyplot as plt
from collections import deque
import copy
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Player:
    def __init__(self, name: str, turn: int, test: bool) -> None:
        self.name = name
        self.turn = turn # -1|1
        self.test = test

    def move(self, board_arr) -> int:
        raise NotImplementedError("Implemented by child class.")


class Bot(Player):
    def __init__(self, name: str, turn: int, test: bool) -> None:
        super().__init__(name, turn, test)

    def move(self, piece_arrays) -> int:
        """Make move based on current board state."""
        board_arr = piece_arrays[self.turn] + piece_arrays[-self.turn]
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        available_cols = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        col = random.choice(available_cols)
        return col
    
class RLBot(Bot):
    def __init__(self, name: str, turn: int, test: bool) -> None:
        super().__init__(name, turn, test)

        self.activation = torch.nn.ReLU
        self.model = self.initialize_model()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.gamma = 0.9
        self.epsilon = self.get_epsilon()
        self.epsilon_decay = 0.0006

        self.stop_training = False # early stopping flag
        self.min_loss_dict = {'current_min': np.inf, 'num_steps': 0}
        self.patience = 600 # num. steps
        self.losses = []
        self.rewards = []
        self.n_moves = 0
        self.n_epoch_moves = 0
        #sqars: state_t, q_vals_t, action_t, reward_t, state_t+1
        self.current_sqars = [None, None, None, None, None]
        self.reward_vals = {
            'draw': 5,
            'win': 10,
            'loss': -10,
            'move': -1,
        }

    def get_epsilon(self):
        return 0.0 if (self.turn!=-1 or self.test) else 0.4

    def initialize_model(self):
        input_n = 84
        hidden_n = 200
        hidden_n_2 = 500
        hidden_n_3 = 200
        hidden_n_4 = 50
        output_n = 7

        model = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n),
            self.activation(),
            torch.nn.Linear(hidden_n, hidden_n),
            self.activation(),
            torch.nn.Linear(hidden_n, hidden_n_2),
            self.activation(),
            torch.nn.Linear(hidden_n_2, hidden_n_2),
            self.activation(),
            torch.nn.Linear(hidden_n_2, hidden_n_3),
            self.activation(),
            torch.nn.Linear(hidden_n_3, hidden_n_4),
            self.activation(),
            torch.nn.Linear(hidden_n_4, output_n),
        )
        model.to(device)
        return model
    
    def get_state_array(self, piece_arrays: Dict):
        state_array = np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]])
        return state_array

    def process_state(self, curr_state: np.array):
        num_cells = curr_state[0].shape[0]*curr_state[0].shape[1]
        curr_state = curr_state.reshape(1, num_cells*2)
        if isinstance(self.activation, torch.nn.ReLU):
            curr_state += np.random.rand(1, num_cells*2)/100.0

        state = torch.from_numpy(curr_state).float().to(device)
        return state

    def move(self, piece_arrays: Dict) -> int:
        curr_state = self.get_state_array(piece_arrays)
        state = self.process_state(curr_state)
        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(state)
        q_vals_ = q_vals.data.cpu().numpy()
        
        board_arr = piece_arrays[self.turn] + piece_arrays[-self.turn]
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        if random.random() < self.epsilon:
            action_ = random.choice([i for i,n in enumerate(available_cols) if n])
        else:
            q_vals_ = np.where((q_vals_*available_cols.astype(int))==0, q_vals.min().item()-1, q_vals_)
            action_ = np.argmax(q_vals_)
        self.epsilon *= (1-self.epsilon_decay)

        self.current_sqars = [None, None, None, None, None]
        self.current_sqars[0] = state
        self.current_sqars[1] = q_vals
        self.current_sqars[2] = action_
        self.n_moves += 1; self.n_epoch_moves += 1
        return action_
    
    def get_reward(self, move_result):
        if move_result is None:
            reward = self.reward_vals['move']
        elif 'draw' in move_result:
            reward = self.reward_vals['draw']
        else:
            win_side = int(move_result.split('_')[-1])
            reward = self.reward_vals['win'] if win_side==self.turn else self.reward_vals['loss']
        self.rewards.append(reward)
        return reward
    
    def reset_vars(self):
        self.current_sqars = [None, None, None, None, None]
        self.n_epoch_moves = 0

    def update_early_stopping(self):
        check_window_steps = 50
        if len(self.losses)>=check_window_steps:
            current_avg_loss = np.mean(self.losses[-check_window_steps:])
            if current_avg_loss < self.min_loss_dict['current_min']:
                self.min_loss_dict['current_min'] = current_avg_loss
                self.min_loss_dict['num_steps'] = 0
            else:
                self.min_loss_dict['num_steps'] += 1
            
            if self.min_loss_dict['num_steps']>=self.patience:
                self.stop_training = True
    
    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)

        self.current_sqars[4] = new_state
        self.current_sqars[3] = self.get_reward(result)

        
        # get Q values of new state to update last state's Q values
        with torch.no_grad():
            new_q = self.model(new_state)
        max_q = torch.max(new_q)

        # target value
        Y = reward if result is not None else reward + (self.gamma*max_q)
        Y = torch.Tensor([Y]).detach().squeeze().to(device)
        X = self.current_sqars[1].squeeze()[self.current_sqars[2]]
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

        self.update_early_stopping()
        if result is not None: # game epoch is over
            self.reset_vars()
        return self

    def load_model(self, path):
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()
        if hasattr(self, 'target_model'):
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        return self

    def plot_results(self, show=False, save_path=None):
        """Plots losses, rewards by move"""
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Moves")
        ax1.set_ylabel("Rewards")
        ax1.plot(np.arange(len(self.rewards)), self.rewards, color='r')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss")
        ax1.plot(np.arange(len(self.losses)), self.losses, color='b')

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.clf()

    def save_model_and_results(self, model_path: str):
        # model
        torch.save(self.model.state_dict(), model_path+'model.pth')
        # results: losses, rewards, avg. win rate
        np.savetxt(model_path+'losses.csv', np.array(self.losses), delimiter=',', header='losses')
        self.plot_results(save_path=model_path+'results.png')

class Human(Player):
    def __init__(self, name: str, turn: int, test: bool) -> None:
        super().__init__(name, turn, test)

class RLBotDDQN(RLBot):
    def __init__(self, name: str, turn: int, test: bool) -> None:
        super().__init__(name, turn, test)
        self.memory = deque(maxlen=1000)
        self.batch_size = 200

        self.target_sync_freq = 200
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def reset_self_play(self, turn: int):
        self.lr = 1e-3
        self.epsilon = self.get_epsilon()
        self.losses = []
        self.rewards = []
        self.n_moves = 0
        self.n_epoch_moves = 0
        self.current_sqars = [None, None, None, None, None]
        self.turn = turn

        self.memory = deque(maxlen=1000)

        self.stop_training = False
        self.min_loss_dict = {'current_min': np.inf, 'num_steps': 0}

        self.model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        return self

    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)
        self.current_sqars[4] = new_state

        self.current_sqars[3] = self.get_reward(result)
        curr_experience = (#sqars: state_t, q_vals_t, action_t, reward_t, state_t+1
            self.current_sqars[0], # state t
            self.current_sqars[2], # action t
            self.current_sqars[3], # reward t
            self.current_sqars[4], # state t+1
            int(result is not None)
        )
        self.memory.append(curr_experience)

        if len(self.memory) <= self.batch_size:
            return self

        minibatch = random.sample(self.memory, self.batch_size)
        
        s_batch = torch.cat([s for (s,a,r,s2,d) in minibatch]).to(device)
        a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch]).to(device)
        r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch]).to(device)
        s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch]).to(device)
        d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch]).to(device)

        self.model.train(True)
        self.optimizer.zero_grad()
        q1 = self.model(s_batch)
        # get Q values of new state to update last state's Q values
        with torch.no_grad():
            new_q = self.target_model(s2_batch)
        max_q = torch.max(new_q, dim=1)

        # target value
        Y = r_batch + self.gamma*((1-d_batch)*max_q[0])
        X = q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(X, Y.detach())
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        if self.n_moves % self.target_sync_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        
        self.update_early_stopping()
        if result is not None: # game epoch is over
            self.reset_vars()
        return self

class Connect4Env(gym.Env):
    metadata = {"render_modes": []}
    BOARD_ROWS, BOARD_COLS = 6, 7

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0., high=1.,
            shape=(2 * self.BOARD_ROWS * self.BOARD_COLS,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.BOARD_COLS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0., False, False, {}

class Connect4FeatureExtractor(BaseFeaturesExtractor):
    """84 → 200 → 500 → 200 → 50"""
    def __init__(self, observation_space, features_dim=50):
        super().__init__(observation_space, features_dim)
        n = observation_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 200),   torch.nn.ReLU(),
            torch.nn.Linear(200, 200), torch.nn.ReLU(),
            torch.nn.Linear(200, 500), torch.nn.ReLU(),
            torch.nn.Linear(500, 200), torch.nn.ReLU(),
            torch.nn.Linear(200, features_dim), torch.nn.ReLU(),
        )
    def forward(self, obs):
        return self.net(obs)

class PPOBot(Bot):
    """
    PPO agent using current interface
    """

    def __init__(
        self,
        name: str,
        turn: int,
        test: bool,
        update_freq: int = 256,
        n_epochs: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
    ):
        super().__init__(name, turn, test)

        self.update_freq = update_freq
        self.n_ppo_epochs = n_epochs
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range

        self.epsilon = self.get_epsilon()
        self.epsilon_decay = 0.0004

        policy_kwargs = dict(
            features_extractor_class=Connect4FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=50),
            net_arch=[],
        )
        self.model = PPO(
            policy=ActorCriticPolicy,
            env=Connect4Env(),
            learning_rate=lr,
            n_steps=update_freq,
            batch_size=64,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self._rollout: list = []

        # keep tracking steps
        self.stop_training = False
        self.min_loss_dict = {"current_min": np.inf, "num_steps": 0}
        self.patience = 600
        self.losses: list = []
        self.rewards: list = []
        self.n_moves = 0
        self.n_epoch_moves = 0
        self.current_sqars = [None, None, None, None, None]

        self.reward_vals = {"draw": 5, "win": 10, "loss": -10, "move": -1}

    def get_epsilon(self):
        return 0.25 # 0.0 if (self.turn != -1 or self.test) else

    def _get_obs(self, piece_arrays: Dict) -> np.ndarray:
        return np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]]).reshape(-1).astype(np.float32)

    def _available_mask(self, piece_arrays: Dict) -> np.ndarray:
        board = piece_arrays[self.turn] + piece_arrays[-self.turn]
        return board.sum(axis=0) < board.shape[0]

    def move(self, piece_arrays: Dict) -> int:
        obs  = self._get_obs(piece_arrays)
        mask = self._available_mask(piece_arrays)

        if random.random() < self.epsilon:
            action = random.choice([i for i, ok in enumerate(mask) if ok])
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                logits = dist.distribution.logits.cpu().numpy().squeeze()
            logits[~mask] = logits.min() - 1e6   # suppress illegal moves
            action = int(np.argmax(logits))

        self.epsilon *= (1 - self.epsilon_decay)
        self.current_sqars[0] = obs
        self.current_sqars[2] = action
        self.n_moves += 1
        self.n_epoch_moves += 1
        return action

    def get_reward(self, move_result) -> float:
        if move_result is None:
            r = self.reward_vals["move"]
        elif "draw" in move_result:
            r = self.reward_vals["draw"]
        else:
            win_side = int(move_result.split("_")[-1])
            r = self.reward_vals["win"] if win_side == self.turn else self.reward_vals["loss"]
        self.rewards.append(r)
        return r

    def train(self, new_piece_arrays: Dict, result):
        if self.test:
            return self

        new_obs = self._get_obs(new_piece_arrays)
        reward  = self.get_reward(result)
        done    = result is not None

        obs_t = torch.tensor(self.current_sqars[0], dtype=torch.float32).unsqueeze(0).to(self.model.policy.device)
        act_t = torch.tensor([[self.current_sqars[2]]], dtype=torch.long).to(self.model.policy.device)
        with torch.no_grad():
            values, log_probs, _ = self.model.policy.evaluate_actions(obs_t, act_t)

        self._rollout.append((
            self.current_sqars[0], self.current_sqars[2],
            reward, done, values.item(), log_probs.item(), new_obs
        ))

        if len(self._rollout) >= self.update_freq:
            self.losses.append(self._ppo_update())
            self._update_early_stopping()
            self._rollout = []

        if done:
            self.reset_vars()
        return self

    def _ppo_update(self) -> float:
        T, device = len(self._rollout), self.model.policy.device
        obs_arr  = torch.tensor(np.array([x[0] for x in self._rollout]), dtype=torch.float32).to(device)
        acts_arr = torch.tensor([x[1] for x in self._rollout], dtype=torch.long).to(device)
        rews_arr = np.array([x[2] for x in self._rollout], dtype=np.float32)
        done_arr = np.array([x[3] for x in self._rollout], dtype=np.float32)
        vals_arr = np.array([x[4] for x in self._rollout], dtype=np.float32)
        lps_arr  = torch.tensor([x[5] for x in self._rollout], dtype=torch.float32).to(device)
        next_obs = torch.tensor(self._rollout[-1][6], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            last_val = self.model.policy.predict_values(next_obs).item()
        advantages, gae = np.zeros(T, dtype=np.float32), 0.
        for t in reversed(range(T)):
            nv    = last_val if t == T - 1 else vals_arr[t + 1]
            delta = rews_arr[t] + self.gamma * nv * (1 - done_arr[t]) - vals_arr[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - done_arr[t]) * gae
            advantages[t] = gae
        returns    = advantages + vals_arr
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(device)

        total, n = 0., 0
        for _ in range(self.n_ppo_epochs):
            for idx in [np.random.permutation(T)[s:s+64] for s in range(0, T, 64)]:
                new_vals, new_lps, entropy = self.model.policy.evaluate_actions(
                    obs_arr[idx], acts_arr[idx]
                )
                ratio = torch.exp(new_lps - lps_arr[idx])
                policy_loss = -torch.min(
                    ratio * adv_t[idx],
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_t[idx]
                ).mean()
                value_loss   = 0.5 * ((new_vals.squeeze() - ret_t[idx]) ** 2).mean()
                entropy_loss = -0.01 * entropy.mean()
                loss = policy_loss + value_loss + entropy_loss

                self.model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
                self.model.policy.optimizer.step()
                total += loss.item(); n += 1
        return total / max(n, 1)

    # not implemented, ignore
    def _update_early_stopping(self):
        w = 50
        if len(self.losses) >= w:
            avg = np.mean(self.losses[-w:])
            if avg < self.min_loss_dict["current_min"]:
                self.min_loss_dict["current_min"] = avg
                self.min_loss_dict["num_steps"] = 0
            else:
                self.min_loss_dict["num_steps"] += 1
            if self.min_loss_dict["num_steps"] >= self.patience:
                self.stop_training = True

    def reset_vars(self):
        self.current_sqars = [None, None, None, None, None]
        self.n_epoch_moves = 0

    def reset_self_play(self, turn: int):
        self.turn = turn
        self.epsilon = self.get_epsilon()
        self.losses = []; self.rewards = []
        self.n_moves = 0; self.n_epoch_moves = 0
        self.current_sqars = [None, None, None, None, None]
        self._rollout = []
        self.stop_training = False
        self.min_loss_dict = {"current_min": np.inf, "num_steps": 0}

        return self

    def save_model_and_results(self, model_path: str):
        torch.save(self.model.policy.state_dict(), model_path + "model.pth")
        np.savetxt(model_path + "ppo_losses.csv", np.array(self.losses),
                   delimiter=",", header="losses")
        self.plot_results(save_path=model_path + "ppo_results.png")

    def load_model(self, path):
        state = torch.load(path, map_location=self.model.policy.device)
        self.model.policy.load_state_dict(state)
        self.model.policy.set_training_mode(False)
        return self

    def plot_results(self, show=False, save_path=None):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Moves");  ax1.set_ylabel("Rewards", color="r")
        ax1.plot(np.arange(len(self.rewards)), self.rewards, color="r", alpha=0.5)
        ax2 = ax1.twinx();        ax2.set_ylabel("Loss", color="b")
        ax2.plot(np.arange(len(self.losses)),  self.losses,  color="b", alpha=0.7)
        fig.tight_layout()
        if save_path: plt.savefig(save_path)
        if show:      plt.show()
        plt.clf()