import torch
import torch.nn as nn
import time
import numpy as np

class Config():
    """
    用于配置整个算法的参数
    """
    def __init__(self,max_episode=2000,max_steps=100,learning_rate=0.01,gamma=0.99,epsilon=0.1,DBM=True):
        # 训练参数
        self.max_episode = max_episode # 训练的次数
        self.max_steps = max_steps # 每次训练的最大步数
        self.learning_rate = learning_rate # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # 探索率
        # 调试参数
        self.DBM = DBM
    
    def print_cfg(self):
        print("-"*3,"config","-"*3)
        print("max_episode:",self.max_episode)
        print("max_steps:",self.max_steps)
        print("learning_rate:",self.learning_rate)
        print("gamma:",self.gamma)
        print("epsilon:",self.epsilon)
        print("-"*12)

import inspect
def print_(*args):
    """
    增强版print函数，自动打印变量名和值
    注意这里不能打印临时变量。
    示例:
        x = 10
        print_(x)  # 输出: x: 10
    """
    frame = inspect.currentframe()
    try:
        # 获取调用帧的局部变量
        caller_locals = frame.f_back.f_locals
        
        for arg in args:
            # 查找变量名
            var_name = [name for name, value in caller_locals.items() if value is arg][0]
            print(f"{var_name}: {arg}")
    except:
        print(*args)
    finally:
        del frame  # 避免内存泄漏

def save_data(rewards = [], filename = "xxxrewards.pkl"):
    """
    save rewards to 1data_saved
    :param: rewards: list of rewards
    :param: filename: str, filename to save
    """
    import pickle
    with open(f"1data_saved/{filename}", "wb") as f:
        pickle.dump(rewards, f)

def save_model(model = None, filename = "xxxmodel.pth"):
    if model is None:
        print("Model is none, cannot be saved!")
    else:
        torch.save(model.state_dict(),f"2model_trained/{filename}")

##########################
# for DQN using
class SimpleNN(nn.Module):
    def __init__(self,input_size=4,output_size=2,hidden_size=128,num_layers=2):
        """
        Construct for DQN here.
        """
        super(SimpleNN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss() # 评估target和q的差异
        
        assert self.num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size,self.hidden_size))
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size,self.hidden_size))
        self.layers.append(nn.Linear(hidden_size,output_size))
        
    def forward(self,x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x

from collections import deque
import random
class ReplayMemory():
    def __init__(self, capacity = 200):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        """
        :param: state: in numpy is fine
        """
        self.memory.append((state, action, reward, next_state))
    def sample(self, batch_size = 30):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Epsilon():
    def __init__(self, original_val = 0.01, gamma = 1, min_val = 1e-5):
        self.epsilon = original_val
        self.gamma = gamma
        self.min_val = min_val
    def update(self):
        self.epsilon = max(self.epsilon * self.gamma, self.min_val)

def select_action(state, env, model, epsilon = None, config = None, device="cpu"):
    """
    整合了类型转换、epsilon-greedy，以及model前向传播。
    :param: state: numpy数组，来自env的直接输出。
    :return: action:int，来自模型的输出。
    """
    if np.random.rand() < epsilon.epsilon:
        action = env.action_space.sample()
    else:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # 添加一个新的维度，作为batchsize，在维度0处
        if config != None and config.DBM == True:
            print_(state)
            print_(model(state))
            print_(model(state).argmax())
        action = model(state).argmax().item()
        if config != None and config.DBM == True:
            print_(action)
    epsilon.update()
    return action

def update_model(model, target_model, memory, optimizer, gamma = 0.99, device = "cpu", config = None, batchsize = 30, total_step_num = 0, tau = 0.005):
    """
    整合了模型更新、损失计算、反向传播、参数更新。
    根据课本：可以在智能体没执行一个动作后，对w做几次更新；也可以在没完成一个回合后，对w做几次更新。
    :param: model: 模型
    :param: memory: 经验回放池；[(s_t,a_t,r_t,s_t+1)]
    :param: optimizer: 优化器
    :param: gamma: 折扣因子
    :param: device: 设备
    :param: batchsize: 批次大小
    :return: loss: 损失
    """
    # 从经验回放池中随机采样
    batch = memory.sample(batchsize)
    s_t,a_t,r_t,s_t_1 = zip(*batch) # ???????????????????? 
    s_t = torch.from_numpy(np.array(s_t)).float().to(device)
    a_t = torch.from_numpy(np.array(a_t)).long().to(device)
    r_t = torch.from_numpy(np.array(r_t)).float().to(device)
    s_t_1 = torch.from_numpy(np.array(s_t_1)).float().to(device)

    # 更target_network
    if total_step_num == 1:
        target_model.load_state_dict(model.state_dict())
    elif total_step_num % 200 == 0:
        target_state_dict = target_model.state_dict()
        state_dict = model.state_dict()
        for key in target_state_dict:
            target_state_dict[key] = tau*state_dict[key] + (1-tau)*target_state_dict[key]
        target_model.load_state_dict(target_state_dict)
    # 计算TD目标值
    with torch.no_grad(): # 也就是说这一部分不计算梯度，也不更新参数，单纯只是得到一个结果
        target = r_t + gamma * target_model(s_t_1).max(dim=1)[0] # 注意这里的写法啊，max(dim=1)因为0是batch，这里只有一个；然后最后的target也是一个tensor，[0]表示值，[1]表示indice
    if sum(target) > 1e5:
        print("warning! Q is too large!")
    # 计算Q值
    q = model(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1) # 注意这里需要使用切片操作保留batchsize这个维度。
    if config.DBM == True:
        print_(target)
        print_(model(s_t))
        print_(q)
    # 计算损失
    loss = model.loss(q, target)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1)
    optimizer.step()
    # 打印损失
    if config.DBM == True:
        print_(loss)
    return 1

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward