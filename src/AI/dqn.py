# Rainbow-DQN (C51 + Dueling + Noisy + PER + Double + N-step)

from dataclasses import dataclass
from typing import Tuple, Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------- Noisy Linear (Factorized) --------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # As in Fortunato et al.
        self.weight_sigma.data.fill_(self.sigma0 / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / np.sqrt(self.in_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return torch.addmm(b, x, w.t())


# -------- C51 Dueling Network --------
class CategoricalDuelingDQN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden: int,
                 atoms: int, vmin: float, vmax: float):
        super().__init__()
        self.num_actions = num_actions
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))

        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )

        # Dueling heads with Noisy layers (output prob logits per action over atoms)
        self.advantage = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, num_actions * atoms),
        )
        self.value = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits of shape [B, A, atoms]
        f = self.feature(x)
        adv = self.advantage(f).view(-1, self.num_actions, self.atoms)
        val = self.value(f).view(-1, 1, self.atoms)
        # Dueling combine (logits level, subtract mean advantage for stability)
        adv_mean = adv.mean(dim=1, keepdim=True)
        logits = val + (adv - adv_mean)
        return logits

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        # probability distributions per action over atoms
        logits = self.forward(x)
        return torch.softmax(logits, dim=2)  # [B, A, atoms]

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        # expectation over atoms
        probs = self.dist(x)  # [B, A, atoms]
        q = torch.sum(probs * self.support, dim=2)
        return q  # [B, A]

    def reset_noise(self):
        # reset all NoisyLinear layers
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# -------- Prioritized Replay (SumTree) --------
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.ptr = 0
        self.size = 0

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # Propagate change
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = 0
        # Traverse the tree
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity - 1 + 1  # simplify: idx - (capacity-1)
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PERBuffer:
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 1e-6, eps: float = 1e-6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), float(done))
        priority = self.max_priority
        self.tree.add(priority, data)

    def sample(self, batch_size: int):
        # Ensure we have enough data
        assert self.tree.size >= batch_size, "Not enough samples in buffer."

        batch = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        idxs = []
        priorities = []

        total = self.tree.total
        # 防止 total 为 0 的极端情况（理论上有数据时不应为 0）
        segment = (total / batch_size) if total > 0 else 1.0

        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment) if total > 0 else np.random.uniform(0.0, 1.0)
            idx, p, data = self.tree.get(s)

            # 如果取到未填充叶子（data 为 None）或优先级为 0，则从已填充区间随机挑一个样本
            if (data is None) or (p <= 0.0):
                # 只在 [0, size) 中挑选有效 data_index
                data_index = np.random.randint(0, self.tree.size)
                idx = data_index + (self.tree.capacity - 1)
                p = float(self.tree.tree[idx])
                data = self.tree.data[data_index]

            # 这里应当保证 data 一定有效
            state, action, reward, next_state, done = data
            for k, v in zip(batch.keys(), [state, action, reward, next_state, done]):
                batch[k].append(v)
            idxs.append(idx)
            priorities.append(p)

        # importance-sampling weights
        probs = np.array(priorities, dtype=np.float32) / max(self.tree.total, 1e-8)
        N = max(1, self.tree.size)
        weights = (N * probs) ** (-self.beta)
        weights /= (weights.max() + 1e-8)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = np.stack(batch['state'])
        actions = np.array(batch['action'], dtype=np.int64)
        rewards = np.array(batch['reward'], dtype=np.float32)
        next_states = np.stack(batch['next_state'])
        dones = np.array(batch['done'], dtype=np.float32)

        return (states, actions, rewards, next_states, dones, idxs, weights)

    def update_priorities(self, idxs: List[int], priorities: np.ndarray):
        priorities = np.abs(priorities) + self.eps
        priorities = np.power(priorities, self.alpha)
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, float(p))
            self.max_priority = max(self.max_priority, float(p))


# -------- N-step helper --------
class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
        return self._pop_ready(d)

    def _pop_ready(self, done_flag: bool):
        ready = []
        while len(self.buf) >= self.n or (done_flag and len(self.buf) > 0):
            R = 0.0
            discount = 1.0
            for i in range(min(self.n, len(self.buf))):
                R += self.buf[i][2] * discount
                discount *= self.gamma
                if self.buf[i][4]:
                    # early termination within n steps
                    break
            s, a = self.buf[0][0], self.buf[0][1]
            ns = self.buf[min(self.n - 1, len(self.buf) - 1)][3]
            d = any(x[4] for x in list(self.buf)[:self.n])
            ready.append((s, a, R, ns, d))
            self.buf.popleft()
            if d:
                self.buf.clear()
        return ready


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 256
    target_update: int = 1000
    epsilon_start: float = 1.0   # 保留字段（不再使用，兼容命令行）
    epsilon_end: float = 0.05    # 保留字段（不再使用，兼容命令行）
    epsilon_decay_steps: int = 50_000  # 保留字段（不再使用，兼容命令行）
    replay_capacity: int = 200_000

    # Rainbow 相关
    atoms: int = 51
    vmin: float = 0.0
    vmax: float = 4096.0
    hidden: int = 256
    n_steps: int = 3
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-6
    per_eps: float = 1e-6
    noisy_sigma0: float = 0.5
    grad_clip: float = 10.0


class DQNAgent:
    """
    Rainbow-DQN Agent（保留类名以兼容 train.py）
    """
    def __init__(self, state_dim=16, num_actions=4, device="cpu", config: DQNConfig = DQNConfig()):
        self.device = torch.device(device)
        self.cfg = config
        self.num_actions = num_actions

        self.q = CategoricalDuelingDQN(
            input_dim=state_dim,
            num_actions=num_actions,
            hidden=config.hidden,
            atoms=config.atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        ).to(self.device)

        self.q_target = CategoricalDuelingDQN(
            input_dim=state_dim,
            num_actions=num_actions,
            hidden=config.hidden,
            atoms=config.atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        ).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=config.lr)
        self.loss_fn = nn.KLDivLoss(reduction="none")  # 用于分布式损失（我们将手动做交叉熵）

        self.buffer = PERBuffer(
            capacity=config.replay_capacity,
            state_dim=state_dim,
            alpha=config.per_alpha,
            beta=config.per_beta,
            beta_increment=config.per_beta_increment,
            eps=config.per_eps,
        )
        self.nstep = NStepBuffer(n=config.n_steps, gamma=config.gamma)

        self.train_steps = 0
        self.delta_z = (config.vmax - config.vmin) / (config.atoms - 1)
        # 预备支持向量（放到设备上）
        self.support = torch.linspace(config.vmin, config.vmax, config.atoms, device=self.device)

    # 兼容旧接口（NoisyNet 下不使用 epsilon，但保留方法）
    def epsilon(self) -> float:
        return 0.0

    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        if eval_mode:
            self.q.eval()
        else:
            self.q.train()
            # 每次采样都刷新噪声
            self.q.reset_noise()

        q = self.q.q_values(s)  # [1, A]
        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, ns, d):
        # N-step 聚合并放入 PER
        ready = self.nstep.push(s, a, r, ns, d)
        for (ss, aa, Rn, nss, dd) in ready:
            self.buffer.push(ss, aa, Rn, nss, dd)

    def _project_distribution(self, rewards: torch.Tensor, dones: torch.Tensor, next_dist: torch.Tensor, gamma_n: float) -> torch.Tensor:
        # next_dist: [B, atoms] (选取了 next_action 后的分布)
        batch_size = rewards.size(0)
        vmin, vmax, atoms = self.cfg.vmin, self.cfg.vmax, self.cfg.atoms
        delta_z = self.delta_z
        support = self.support.view(1, -1)  # [1, atoms]

        # Tz
        Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * support  # [B, atoms]
        Tz = Tz.clamp(min=vmin, max=vmax)

        b = (Tz - vmin) / delta_z  # [B, atoms]
        l = b.floor().clamp(0, atoms - 1)
        u = b.ceil().clamp(0, atoms - 1)

        l_long = l.long()
        u_long = u.long()

        # Distribute probability mass
        m = torch.zeros(batch_size, atoms, device=self.device)
        offset = torch.linspace(0, (batch_size - 1) * atoms, batch_size, device=self.device).unsqueeze(1).long()

        # Add to lower
        m.view(-1).index_add_(0, (l_long + offset).view(-1), (next_dist * (u - b)).view(-1))
        # Add to upper
        m.view(-1).index_add_(0, (u_long + offset).view(-1), (next_dist * (b - l)).view(-1))
        return m

    def learn(self):
        if self.buffer.tree.size < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones, idxs, weights = self.buffer.sample(self.cfg.batch_size)

        s = torch.from_numpy(states).float().to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        r = torch.from_numpy(rewards).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        d = torch.from_numpy(dones).float().to(self.device)
        w = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(self.device)

        # 当前分布（选择 a）
        self.q.train()
        self.q.reset_noise()
        logits = self.q(s)  # [B, A, atoms]
        log_prob = torch.log_softmax(logits, dim=2)
        chosen_log_prob = log_prob.gather(
            1, a.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.cfg.atoms)
        ).squeeze(1)  # [B, atoms]
        # 移除未使用的计算，避免不必要的图分支
        # prob = torch.softmax(logits, dim=2)
        # chosen_prob = prob.gather(1, a.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.cfg.atoms)).squeeze(1)

        with torch.no_grad():
            # Double DQN: 用在线网络选动作
            # 重要：不要在反向传播前重置 NoisyNet 噪声，避免修改计算图依赖的 buffer
            next_q = self.q.q_values(ns)  # [B, A]
            next_action = next_q.argmax(dim=1, keepdim=True)  # [B, 1]

            # 目标网络给出分布
            self.q_target.eval()
            next_dist_all = self.q_target.dist(ns)  # [B, A, atoms]
            next_dist = next_dist_all.gather(
                1, next_action.unsqueeze(-1).expand(-1, -1, self.cfg.atoms)
            ).squeeze(1)  # [B, atoms]
            next_dist = next_dist.clamp(min=1e-6)
            next_dist = next_dist / next_dist.sum(dim=1, keepdim=True)

            gamma_n = self.cfg.gamma ** self.cfg.n_steps
            target_dist = self._project_distribution(r, d, next_dist, gamma_n)  # [B, atoms]
            target_dist = target_dist.clamp(min=1e-6)
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)

        # 分布式交叉熵损失：-sum target * log(pred)
        per_sample_loss = -(target_dist * chosen_log_prob).sum(dim=1)  # [B]
        loss = (w * per_sample_loss).mean()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.cfg.grad_clip)
        self.optim.step()

        # 更新 PER priority
        prios = per_sample_loss.detach().cpu().numpy()
        self.buffer.update_priorities(idxs, prios)

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        # 训练后再重置噪声（此时图已释放，安全）
        self.q.reset_noise()
        self.q_target.reset_noise()

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "q_state_dict": self.q.state_dict(),
            "target_state_dict": self.q_target.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_target.load_state_dict(ckpt.get("target_state_dict", ckpt["q_state_dict"]))