import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time

train_start = time.time()  # 训练开始时间
# 设定是否使用CUDA（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 网络构建
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义第一层全连接层：输入特征784，输出特征128
        self.fc1 = nn.Linear(784, 128)
        # 定义第二层全连接层：输入特征128，输出特征128
        self.fc2 = nn.Linear(128, 128)
        # 定义输出层：输入特征128，输出特征10（对应10个数字类别）
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # 对第一层的输出应用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 对第二层的输出也应用ReLU激活函数
        x = torch.relu(self.fc2(x))
        # 通过输出层得到最终的分类结果
        x = self.output(x)
        return x

# 数据加载与处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片数据转换为Tensor：自动将图像数据从H×W×C转换为C×H×W格式；将像素值从[0,255]缩放到[0,1]范围
    transforms.Normalize((0.5,), (0.5,)),  # 对数据进行归一化处理：输入范围[0,1]会被转换为[(0-0.5)/0.5, (1-0.5)/0.5] = [-1,1]
])

# 加载训练数据集
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 加载测试数据集
test_set = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 将模型实例化并转移到定义的设备（CPU或GPU）
model = SimpleNN().to(device)

# 选择优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练过程
for epoch in range(5):  # 训练5个循环周期
    for images, labels in train_loader:
        # 调整图片形状并转移到相同的设备，全连接层要求输入是二维的[batch_size, features]
        images, labels = images.view(-1, 28*28).to(device), labels.to(device) # 将图像从[batchsize, 1, 28, 28]转换为[batchsize(-1表示自动计算), 28*28]
        optimizer.zero_grad()  # 清除历史梯度
        output = model(images)  # 前向传播计算模型输出
        loss = criterion(output, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()  # 将模型设置为评估模式
    total_correct = 0
    total = 0
    with torch.no_grad():  # 禁止梯度计算
        for images, labels in data_loader:
            # 将图片和标签数据转移到相同的设备
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
            output = model(images)  # 前向传播得到预测结果
            _, predicted = torch.max(output.data, 1)  # 得到预测的类别
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()  # 统计正确预测的数量
    # 打印准确率
    print(f'Accuracy: {100 * total_correct / total:.2f}%')

# 使用测试数据集评估模型性能
evaluate_model(model, test_loader)
train_end = time.time()  # 训练结束时间
print("训练时间: {:.2f}秒".format(train_end - train_start))