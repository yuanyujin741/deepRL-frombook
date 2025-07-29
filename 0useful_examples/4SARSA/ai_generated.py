import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class SimpleNN(nn.Module):
    def __init__(self, input_size=4, output_size=2, hidden_size=128, num_layers=2):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout

        assert self.num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)  # 应用 Dropout
        for layer in self.layers[-1:]:
            x = layer(x)
        return x

class NNSARSAAgent:
    def __init__(self, n_actions, n_features, learning_rate=0.001, reward_decay=0.9, e_greedy=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # 初始化 Q 网络和目标网络
        self.q_network = SimpleNN(input_size=n_features, output_size=n_actions)
        self.target_network = SimpleNN(input_size=n_features, output_size=n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 复制初始权重

        # 定义优化器 (例如 Adam)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        # 定义损失函数 (MSELoss)
        self.criterion = nn.MSELoss()

        # 目标网络更新频率
        self.update_counter = 0
        self.TARGET_UPDATE_FREQ = 100

    def choose_action(self, observation):
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        return action

    def predict(self, observation):
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.numpy().squeeze()

    def learn(self, s, a, r, s_, a_, done):
        s_tensor = torch.FloatTensor(s)
        s_next_tensor = torch.FloatTensor(s_)
        r_tensor = torch.FloatTensor([r])

        # 计算当前状态-动作对的 Q 值
        q_current_all = self.q_network(s_tensor.unsqueeze(0))
        q_current = q_current_all[:, a]

        # 使用目标网络计算目标 Q 值
        with torch.no_grad():
            if done:
                q_target = r_tensor
            else:
                q_next_all = self.target_network(s_next_tensor.unsqueeze(0))
                q_next = q_next_all[:, a_]
                q_target = r_tensor + self.gamma * q_next

        # 计算损失
        loss = self.criterion(q_current, q_target)

        # 执行梯度下降步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.update_counter % self.TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.update_counter += 1

def main():
    env = gym.make('CartPole-v1')
    seed = 42
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]

    agent = NNSARSAAgent(n_actions, n_features, learning_rate=0.001, reward_decay=0.9, e_greedy=0.01)

    num_episodes = 5000  # 增加训练 episode 数量
    all_episode_rewards = []
    smoothed_rewards = deque(maxlen=10)
    avg_rewards_last_10 = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward

        all_episode_rewards.append(total_reward)
        smoothed_rewards.append(total_reward)
        if len(smoothed_rewards) == 10:
            avg_reward = sum(smoothed_rewards) / len(smoothed_rewards)
            avg_rewards_last_10.append(avg_reward)
        elif episode > 0:
            avg_reward = sum(smoothed_rewards) / len(smoothed_rewards)
            avg_rewards_last_10.append(avg_reward)
        else:
            avg_rewards_last_10.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_episode_rewards, alpha=0.3, label='Episode Reward (alpha=0.3)', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Deep SARSA on CartPole-v1: Episode Rewards')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    episodes_for_avg = list(range(len(avg_rewards_last_10)))
    plt.plot(episodes_for_avg, avg_rewards_last_10, 'r-', label='Avg Reward (last 10 episodes)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Deep SARSA on CartPole-v1: Smoothed Rewards')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()