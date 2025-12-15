# 强化学习完整教程 (Python)

## 目录
1. [强化学习基础概念](#1-强化学习基础概念)
2. [环境搭建](#2-环境搭建)
3. [多臂老虎机问题](#3-多臂老虎机问题)
4. [马尔可夫决策过程](#4-马尔可夫决策过程)
5. [动态规划方法](#5-动态规划方法)
6. [蒙特卡洛方法](#6-蒙特卡洛方法)
7. [时序差分学习 (Q-Learning)](#7-时序差分学习-q-learning)
8. [深度Q网络 (DQN)](#8-深度q网络-dqn)
9. [策略梯度方法](#9-策略梯度方法)

---

## 1. 强化学习基础概念

### 1.1 什么是强化学习？

强化学习是机器学习的一个分支，通过智能体(Agent)与环境(Environment)的交互来学习最优策略。核心要素包括：

- **智能体(Agent)**: 学习和决策的主体
- **环境(Environment)**: 智能体所处的外部世界
- **状态(State)**: 环境的当前情况
- **动作(Action)**: 智能体可以执行的操作
- **奖励(Reward)**: 环境对智能体动作的反馈
- **策略(Policy)**: 从状态到动作的映射

### 1.2 强化学习的目标

最大化累积奖励(Return): 
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ γ^k R_{t+k+1}
```

其中 γ (gamma) 是折扣因子，范围在 [0,1]，用于平衡即时奖励和未来奖励。

---

## 2. 环境搭建

### 2.1 安装必要的库

```bash
pip install numpy matplotlib gym
pip install torch torchvision  # 用于深度强化学习
```

### 2.2 基础导入

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
```

---

## 3. 多臂老虎机问题

多臂老虎机是强化学习最简单的问题，用于理解探索与利用的权衡。

### 3.1 问题定义

```python
class MultiArmedBandit:
    """多臂老虎机环境"""
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        # 每个臂的真实期望奖励(未知)
        self.true_values = np.random.randn(n_arms)
        
    def pull(self, arm):
        """拉动某个臂，返回奖励"""
        # 奖励 = 真实期望 + 噪声
        reward = self.true_values[arm] + np.random.randn()
        return reward
```

### 3.2 ε-贪心算法实现

```python
class EpsilonGreedy:
    """ε-贪心策略"""
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # 估计的价值
        self.action_counts = np.zeros(n_arms)  # 每个动作的执行次数
        
    def select_action(self):
        """选择动作"""
        if np.random.random() < self.epsilon:
            # 探索: 随机选择
            return np.random.randint(self.n_arms)
        else:
            # 利用: 选择当前最优
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        """更新价值估计"""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        # 增量式更新: Q_new = Q_old + (1/n) * (reward - Q_old)
        self.q_values[action] += (reward - self.q_values[action]) / n

# 运行实验
def run_bandit_experiment(n_steps=1000, n_runs=100):
    """运行多臂老虎机实验"""
    avg_rewards = np.zeros(n_steps)
    
    for run in range(n_runs):
        bandit = MultiArmedBandit(n_arms=10)
        agent = EpsilonGreedy(n_arms=10, epsilon=0.1)
        
        for step in range(n_steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            avg_rewards[step] += reward
    
    avg_rewards /= n_runs
    
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Multi-Armed Bandit Performance')
    plt.grid(True)
    plt.show()
    
    return avg_rewards

# 运行实验
# rewards = run_bandit_experiment()
```

---

## 4. 马尔可夫决策过程

### 4.1 网格世界环境

```python
class GridWorld:
    """简单的网格世界环境"""
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)  # 起始位置
        self.goal = (size-1, size-1)  # 目标位置
        
    def reset(self):
        """重置环境"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """
        执行动作
        action: 0=上, 1=右, 2=下, 3=左
        返回: (next_state, reward, done)
        """
        x, y = self.state
        
        # 执行动作
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 右
            y = min(self.size - 1, y + 1)
        elif action == 2:  # 下
            x = min(self.size - 1, x + 1)
        elif action == 3:  # 左
            y = max(0, y - 1)
        
        self.state = (x, y)
        
        # 计算奖励
        if self.state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # 每步小的负奖励，鼓励快速到达目标
            done = False
        
        return self.state, reward, done
    
    def get_valid_actions(self):
        """获取所有可能的动作"""
        return [0, 1, 2, 3]
```

---

## 5. 动态规划方法

### 5.1 价值迭代算法

```python
class ValueIteration:
    """价值迭代算法"""
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}  # 状态价值函数
        
    def get_action_value(self, state, action):
        """计算状态-动作价值"""
        env = GridWorld(self.env.size)
        env.state = state
        next_state, reward, _ = env.step(action)
        return reward + self.gamma * self.V.get(next_state, 0)
    
    def train(self, max_iterations=1000):
        """训练价值函数"""
        # 初始化所有状态的价值为0
        for i in range(self.env.size):
            for j in range(self.env.size):
                self.V[(i, j)] = 0
        
        for iteration in range(max_iterations):
            delta = 0
            
            # 对每个状态进行更新
            for i in range(self.env.size):
                for j in range(self.env.size):
                    state = (i, j)
                    
                    # 跳过终止状态
                    if state == self.env.goal:
                        continue
                    
                    old_v = self.V[state]
                    
                    # 计算所有动作的价值，选择最大的
                    action_values = [
                        self.get_action_value(state, a) 
                        for a in self.env.get_valid_actions()
                    ]
                    self.V[state] = max(action_values)
                    
                    delta = max(delta, abs(old_v - self.V[state]))
            
            # 如果价值函数收敛，停止迭代
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self.V
    
    def get_policy(self):
        """从价值函数提取策略"""
        policy = {}
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                
                if state == self.env.goal:
                    policy[state] = None
                    continue
                
                # 选择价值最大的动作
                action_values = [
                    self.get_action_value(state, a)
                    for a in self.env.get_valid_actions()
                ]
                policy[state] = np.argmax(action_values)
        
        return policy

# 使用示例
def test_value_iteration():
    env = GridWorld(size=4)
    vi = ValueIteration(env, gamma=0.99)
    
    # 训练
    V = vi.train()
    policy = vi.get_policy()
    
    # 可视化价值函数
    print("\n状态价值函数:")
    for i in range(env.size):
        for j in range(env.size):
            print(f"{V[(i,j)]:6.2f}", end=" ")
        print()
    
    # 可视化策略
    print("\n策略 (0=上, 1=右, 2=下, 3=左):")
    action_symbols = ['↑', '→', '↓', '←']
    for i in range(env.size):
        for j in range(env.size):
            if policy[(i,j)] is None:
                print(" G", end=" ")
            else:
                print(f" {action_symbols[policy[(i,j)]]}", end=" ")
        print()

# test_value_iteration()
```

---

## 6. 蒙特卡洛方法

蒙特卡洛方法通过采样完整的episode来学习价值函数。

```python
class MonteCarloAgent:
    """蒙特卡洛强化学习智能体"""
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(4))  # Q(s,a)
        self.returns = defaultdict(list)  # 存储每个状态-动作对的回报
        
    def select_action(self, state):
        """ε-贪心策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self):
        """生成一个完整的episode"""
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
    def train(self, n_episodes=1000):
        """训练智能体"""
        episode_rewards = []
        
        for episode_num in range(n_episodes):
            episode = self.generate_episode()
            
            # 计算回报
            G = 0
            episode_reward = 0
            
            # 从后向前遍历episode
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                episode_reward += reward
                G = reward + self.gamma * G
                
                # 首次访问MC: 只在状态-动作对首次出现时更新
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
            
            episode_rewards.append(episode_reward)
            
            # 每100个episode打印一次进度
            if (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode_num + 1}, Avg Reward: {avg_reward:.3f}")
        
        return episode_rewards

# 使用示例
def test_monte_carlo():
    env = GridWorld(size=4)
    agent = MonteCarloAgent(env, gamma=0.99, epsilon=0.1)
    
    rewards = agent.train(n_episodes=1000)
    
    # 可视化学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (over 100 episodes)')
    plt.title('Monte Carlo Learning')
    plt.grid(True)
    plt.show()

# test_monte_carlo()
```

---

## 7. 时序差分学习 (Q-Learning)

Q-Learning是最流行的强化学习算法之一，结合了蒙特卡洛和动态规划的优点。

```python
class QLearningAgent:
    """Q-Learning智能体"""
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.Q = defaultdict(lambda: np.zeros(4))  # Q表
        
    def select_action(self, state, training=True):
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, n_episodes=1000):
        """训练智能体"""
        episode_rewards = []
        episode_lengths = []
        
        for episode_num in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:  # 限制最大步数
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # Q-Learning更新
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                state = next_state
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # 衰减探索率
            if episode_num % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # 打印进度
            if (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode_num + 1}: Avg Reward: {avg_reward:.3f}, "
                      f"Avg Length: {avg_length:.1f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards, episode_lengths
    
    def get_policy(self):
        """提取策略"""
        policy = {}
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                policy[state] = np.argmax(self.Q[state])
        return policy

# 使用示例
def test_q_learning():
    env = GridWorld(size=4)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0)
    
    # 训练
    rewards, lengths = agent.train(n_episodes=500)
    
    # 可视化学习曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Q-Learning: Reward Progress')
    ax1.grid(True)
    
    ax2.plot(np.convolve(lengths, np.ones(50)/50, mode='valid'))
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Q-Learning: Episode Length')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 显示学到的策略
    policy = agent.get_policy()
    print("\n学到的策略:")
    action_symbols = ['↑', '→', '↓', '←']
    for i in range(env.size):
        for j in range(env.size):
            print(f" {action_symbols[policy[(i,j)]]}", end=" ")
        print()

# test_q_learning()
```

---

## 8. 深度Q网络 (DQN)

DQN使用神经网络来近似Q函数，能够处理高维状态空间。

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 主网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        
    def select_action(self, state, training=True):
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# 用于CartPole环境的示例（需要安装gym）
def train_dqn_cartpole():
    """
    训练DQN在CartPole环境
    需要: pip install gym
    """
    try:
        import gym
        env = gym.make('CartPole-v1')
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = DQNAgent(state_dim, action_dim)
        
        n_episodes = 500
        target_update_freq = 10
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                result = env.step(action)
                next_state, reward, done = result[0], result[1], result[2]
                
                agent.memory.push(state, action, reward, next_state, done)
                agent.update()
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            agent.decay_epsilon()
            
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        # 可视化
        plt.figure(figsize=(10, 5))
        plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('DQN on CartPole')
        plt.grid(True)
        plt.show()
        
    except ImportError:
        print("请安装gym: pip install gym")

# train_dqn_cartpole()
```

---

## 9. 策略梯度方法

策略梯度方法直接优化策略，而不是学习价值函数。

```python
class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    """REINFORCE算法（基础策略梯度）"""
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state):
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def train_episode(self, env):
        """训练一个episode"""
        states, actions, rewards, log_probs = [], [], [], []
        
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        
        # 生成一个episode
        while not done:
            action, log_prob = self.select_action(state)
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        # 计算回报
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 标准化
        
        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # 更新策略
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return sum(rewards)

def train_reinforce():
    """训练REINFORCE算法"""
    try:
        import gym
        env = gym.make('CartPole-v1')
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = REINFORCEAgent(state_dim, action_dim, lr=0.01)
        
        n_episodes = 1000
        episode_rewards = []
        
        for episode in range(n_episodes):
            reward = agent.train_episode(env)
            episode_rewards.append(reward)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}")
        
        # 可视化
        plt.figure(figsize=(10, 5))
        plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('REINFORCE on CartPole')
        plt.grid(True)
        plt.show()
        
    except ImportError:
        print("请安装gym: pip install gym")

# train_reinforce()
```