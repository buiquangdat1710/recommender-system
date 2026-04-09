import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class SigmaK(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.relu(x) ** self.k


class SiLU_Custom(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class CustomBlock(nn.Module):
    def __init__(self, dim, k=2):
        super().__init__()
        
        self.linear = nn.Linear(dim, dim)
        self.sigma_k = SigmaK(k)
        self.silu = SiLU_Custom()

        # learnable alpha
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        main = self.sigma_k(self.linear(x))     # σ_k(ωx + γ)
        skip = self.alpha * self.silu(x)        # α * b(x)
        return main + skip
# =========================
# 1. Dueling Network (giữ nguyên)
# =========================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256, k=2):
        super().__init__()

        # Feature extractor (stack nhiều block giống hình)
        self.input_layer = nn.Linear(state_dim, hidden_dim)

        self.block1 = CustomBlock(hidden_dim, k)
        self.block2 = CustomBlock(hidden_dim, k)

        # 🔥 Dueling heads
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        x = self.block1(x)
        x = self.block2(x)

        V = self.value_head(x)                     # (B,1)
        A = self.advantage_head(x)                 # (B,num_actions)

        Q = V + (A - A.mean(dim=1, keepdim=True))  # Dueling formula
        return Q


# =========================
# 2. Prioritized Replay Buffer (SumTree)
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha          # mức độ dùng priority (0 = uniform, 1 = full priority)
        self.beta = beta            # importance sampling weight
        self.beta_increment = beta_increment
        self.epsilon = epsilon      # tránh priority = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.pos = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.pos + self.capacity - 1
        self.data[self.pos] = data
        self.update(idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def sample(self, batch_size):
        batch_idx = []
        batch_data = []
        weights = []
        segment = self.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._retrieve(0, s)
            batch_idx.append(idx)
            batch_data.append(self.data[idx - self.capacity + 1])
            prob = self.tree[idx] / self.total()
            weight = (self.size * prob) ** (-self.beta)
            weights.append(weight)

        # chuẩn hoá weights
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        return batch_idx, batch_data, torch.FloatTensor(weights)

    def __len__(self):
        return self.size


# =========================
# 3. Agent với PER và recommend có epsilon-greedy
# =========================
class DQNAgent:
    def __init__(self, state_dim, movie_titles,
                 lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=1,
                 target_update=100, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995,
                 device='cpu', per_alpha=0.6, per_beta=0.4):

        self.device = torch.device(device)
        self.movie_titles = movie_titles
        self.num_actions = len(movie_titles)

        self.policy_net = DuelingDQN(state_dim, self.num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha, beta=per_beta)
        self.batch_size = batch_size
        self.gamma = gamma

        self.target_update = target_update
        self.steps = 0

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # =========================
    # 4. Lưu transition (với priority mặc định cao nhất cho mới)
    # =========================
    def store(self, state, action, reward, next_state, done):
        # priority ban đầu = max priority trong buffer (nếu có) hoặc 1.0
        max_priority = self.memory.tree.max() if len(self.memory) > 0 else 1.0
        self.memory.add(max_priority, (state, action, reward, next_state, done))

    # =========================
    # 5. Chọn action (epsilon-greedy)
    # =========================
    def act(self, state, exclude=[]):
        if random.random() < self.epsilon:
            candidates = [i for i in range(self.num_actions) if i not in exclude]
            return random.choice(candidates)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy().flatten()
        q_values[exclude] = -np.inf
        return int(np.argmax(q_values))

    # =========================
    # 6. Cập nhật mạng với PER (Double DQN + importance sampling)
    # =========================
    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        idxs, batch, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = weights.unsqueeze(1).to(self.device)

        # Current Q
        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions)

        with torch.no_grad():
            # Double DQN
            next_q_policy = self.policy_net(next_states)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            max_q_next = next_q_target.gather(1, next_actions)
            target = rewards + (1 - dones) * self.gamma * max_q_next

        # Loss có nhân importance sampling weights
        td_errors = (current_q - target).detach().cpu().numpy().flatten()
        loss = (weights * (current_q - target).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Cập nhật priority cho các mẫu vừa học (dùng TD-error tuyệt đối)
        for i, idx in enumerate(idxs):
            priority = abs(td_errors[i]) + self.memory.epsilon
            self.memory.update(idx, priority ** self.memory.alpha)

        # Cập nhật target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.save_checkpoint('rl_checkpoint.pt')

        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    # =========================
    # 7. Recommend (hỗ trợ epsilon-greedy để khám phá)
    # =========================
    def recommend(self, state, top_k=10, exclude=[]):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

        q_values[exclude] = -np.inf

        # 🔥 epsilon-greedy
        if random.random() < self.epsilon:
            candidates = [i for i in range(self.num_actions) if i not in exclude]
            if len(candidates) >= top_k:
                chosen = random.sample(candidates, top_k)
            else:
                chosen = candidates
            return [self.movie_titles[i] for i in chosen]

        # Greedy
        top_indices = np.argsort(q_values)[::-1][:top_k]
        return [self.movie_titles[i] for i in top_indices]
    
    def save_checkpoint(self, filepath):
        """Lưu toàn bộ trạng thái của agent vào file."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,                     # PrioritizedReplayBuffer (pickle-able)
            'steps': self.steps,
            'epsilon': self.epsilon,
            # Các tham số cấu hình (để kiểm tra tương thích khi load)
            'state_dim': self.policy_net.feature[0].in_features,  # lấy state_dim từ layer đầu
            'num_actions': self.num_actions,
            'movie_titles': self.movie_titles,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'target_update': self.target_update,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
        }
        torch.save(checkpoint, filepath)
        print(f"[RL] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load trạng thái agent từ file, kiểm tra tương thích cơ bản."""
        if not os.path.exists(filepath):
            print(f"[RL] Checkpoint file {filepath} not found, starting fresh.")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)

        # Kiểm tra tương thích về kích thước state và số action
        if (checkpoint.get('state_dim') != self.policy_net.feature[0].in_features or
            checkpoint.get('num_actions') != self.num_actions):
            print("[RL] Checkpoint incompatible (state_dim or num_actions mismatch). Training from scratch.")
            return False

        # Load state dicts
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['memory']
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']

        # Có thể cập nhật lại các tham số khác nếu muốn (giữ nguyên từ __init__ hoặc lấy từ checkpoint)
        # Ở đây ta giữ nguyên cấu hình từ __init__ để tránh sai lệch, ngoại trừ steps và epsilon
        print(f"[RL] Checkpoint loaded from {filepath} (steps={self.steps}, epsilon={self.epsilon:.4f})")
        return True