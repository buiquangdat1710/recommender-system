import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import threading

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x).squeeze(1)
        return q


class OnlineDQN:
    def __init__(self, state_dim, action_dim, movie_titles, movie_features,
                 lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=32,
                 target_update=100, epsilon=0.1, device='cpu'):
        self.device = torch.device(device)
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.steps = 0
        self.epsilon = epsilon

        self.movie_titles = movie_titles
        self.movie_features = torch.FloatTensor(movie_features).to(self.device)  # (num_movies, action_dim)
        self.num_movies = len(movie_titles)

        self.model_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def store_transition(self, state, action_feat, reward, next_state, done):
        with self.memory_lock:
            self.memory.append((state, action_feat, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        with self.memory_lock:
            batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        with self.model_lock:
            # Current Q
            current_q = self.policy_net(states, actions).unsqueeze(1)

            # Compute target Q using target network
            with torch.no_grad():
                # For each next state, compute Q for all movies and take max
                B = next_states.size(0)
                next_states_exp = next_states.unsqueeze(1).expand(B, self.num_movies, -1)  # (B, num_movies, state_dim)
                movie_feats_exp = self.movie_features.unsqueeze(0).expand(B, -1, -1)  # (B, num_movies, action_dim)

                next_states_flat = next_states_exp.reshape(B * self.num_movies, -1)
                movie_feats_flat = movie_feats_exp.reshape(B * self.num_movies, -1)

                q_next_all = self.target_net(next_states_flat, movie_feats_flat)  # (B*num_movies,)
                q_next_all = q_next_all.reshape(B, self.num_movies)  # (B, num_movies)
                max_q_next = q_next_all.max(dim=1)[0].unsqueeze(1)  # (B,1)

                target = rewards + (1 - dones) * self.gamma * max_q_next

            # Loss
            loss = nn.MSELoss()(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def recommend(self, state, n=12, exclude=[]):
        """Return top n movie titles based on Q-values with epsilon-greedy exploration."""
        with self.model_lock:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
            state_exp = state_tensor.expand(self.num_movies, -1)  # (num_movies, state_dim)
            with torch.no_grad():
                q_values = self.policy_net(state_exp, self.movie_features).cpu().numpy()

        # Exclude movies (already watched)
        exclude_indices = [i for i, t in enumerate(self.movie_titles) if t in exclude]
        q_values[exclude_indices] = -np.inf

        # Epsilon-greedy: with prob epsilon, return random movies from remaining
        if random.random() < self.epsilon:
            candidates = [t for i, t in enumerate(self.movie_titles) if i not in exclude_indices]
            if len(candidates) >= n:
                return random.sample(candidates, n)
            else:
                # Not enough candidates, pad with any (should not happen)
                return candidates + random.sample([t for t in self.movie_titles if t not in candidates], n - len(candidates))

        # Greedy: return top n
        top_indices = np.argsort(q_values)[::-1][:n]
        return [self.movie_titles[i] for i in top_indices]