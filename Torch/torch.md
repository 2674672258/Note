# PyTorch从零基础到精通完整教程

## 目录
1. [环境搭建](#1-环境搭建)
2. [Python基础补充](#2-python基础补充)
3. [PyTorch基础](#3-pytorch基础)
4. [张量操作](#4-张量操作)
5. [自动微分机制](#5-自动微分机制)
6. [神经网络基础](#6-神经网络基础)
7. [数据处理](#7-数据处理)
8. [训练神经网络](#8-训练神经网络)
9. [卷积神经网络](#9-卷积神经网络)
10. [循环神经网络](#10-循环神经网络)
11. [Transformer架构](#11-transformer架构)
12. [高级技巧](#12-高级技巧)
13. [项目实战](#13-项目实战)
14. [性能优化](#14-性能优化)
15. [部署与生产](#15-部署与生产)

---

## 1. 环境搭建

### 1.1 安装Python
推荐使用Python 3.8-3.11版本。

**Windows/Mac/Linux:**
```bash
# 从官网下载安装包
# https://www.python.org/downloads/
```

### 1.2 创建虚拟环境
```bash
# 使用venv创建虚拟环境
python -m venv pytorch_env

# 激活虚拟环境
# Windows
pytorch_env\Scripts\activate
# Linux/Mac
source pytorch_env/bin/activate
```

### 1.3 安装PyTorch
访问 https://pytorch.org 选择合适的版本

**CPU版本:**
```bash
pip install torch torchvision torchaudio
```

**GPU版本(CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU版本(CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.4 验证安装
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

### 1.5 安装常用库
```bash
pip install numpy pandas matplotlib scikit-learn jupyter notebook
pip install tqdm tensorboard pillow opencv-python
```

---

## 2. Python基础补充

### 2.1 NumPy快速回顾
```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 数组操作
print(arr.shape)        # (5,)
print(matrix.shape)     # (2, 3)
print(arr + 10)         # 广播机制
print(matrix.T)         # 转置

# 索引和切片
print(arr[1:4])         # [2, 3, 4]
print(matrix[0, :])     # [1, 2, 3]

# 常用函数
print(np.mean(arr))
print(np.std(arr))
print(np.max(arr))
```

### 2.2 面向对象编程
```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x):
        # 前向传播逻辑
        pass
    
    def backward(self, loss):
        # 反向传播逻辑
        pass

# 实例化
model = NeuralNetwork(784, 128, 10)
```

---

## 3. PyTorch基础

### 3.1 什么是PyTorch
PyTorch是一个基于Python的科学计算包,主要特点:
- 灵活的深度学习框架
- 强大的GPU加速
- 动态计算图
- 自动微分
- 丰富的API和工具

### 3.2 PyTorch核心概念
- **Tensor(张量)**: PyTorch的基本数据结构
- **Autograd**: 自动微分引擎
- **nn.Module**: 神经网络模块基类
- **optim**: 优化器
- **DataLoader**: 数据加载器

### 3.3 第一个PyTorch程序
```python
import torch

# 创建张量
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
z = x + y
print(z)  # tensor([5., 7., 9.])

# 矩阵乘法
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.mm(a, b)
print(c.shape)  # torch.Size([2, 4])
```

---

## 4. 张量操作

### 4.1 创建张量
```python
import torch

# 从列表创建
tensor_from_list = torch.tensor([1, 2, 3])

# 创建特殊张量
zeros = torch.zeros(3, 4)           # 全0
ones = torch.ones(2, 3)             # 全1
eye = torch.eye(3)                  # 单位矩阵
random = torch.rand(2, 3)           # [0,1)均匀分布
randn = torch.randn(2, 3)           # 标准正态分布
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# 指定数据类型
float_tensor = torch.tensor([1, 2], dtype=torch.float32)
int_tensor = torch.tensor([1.5, 2.5], dtype=torch.int64)

# 指定设备
cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
```

### 4.2 张量属性
```python
x = torch.randn(2, 3, 4)

print(x.shape)          # torch.Size([2, 3, 4])
print(x.size())         # torch.Size([2, 3, 4])
print(x.dtype)          # torch.float32
print(x.device)         # cpu 或 cuda:0
print(x.requires_grad)  # False
print(x.ndim)           # 3 (维度数)
print(x.numel())        # 24 (元素总数)
```

### 4.3 张量索引和切片
```python
x = torch.randn(4, 5, 6)

# 基本索引
print(x[0])           # 第一个元素 (5, 6)
print(x[0, 1])        # (6,)
print(x[0, 1, 2])     # 标量

# 切片
print(x[:2])          # 前两个 (2, 5, 6)
print(x[:, :3])       # (4, 3, 6)
print(x[..., :2])     # (4, 5, 2)

# 高级索引
indices = torch.tensor([0, 2])
print(x[indices])     # 选择第0和第2个

# 布尔索引
mask = x > 0
print(x[mask])        # 所有正数元素
```

### 4.4 张量变形
```python
x = torch.randn(2, 3, 4)

# reshape - 当数据在内存不连续时，会复制数据返回一个新的张量，比如经过转置、切片等操作后的张量
y = x.reshape(6, 4)
y = x.reshape(-1, 4)  # -1自动计算

# view - 共享内存(需要连续)
y = x.view(6, 4)
y = x.view(-1)        # 展平

# transpose - 转置
y = x.transpose(0, 1)  # 交换维度0和1

# permute - 重排维度
y = x.permute(2, 0, 1)  # (4, 2, 3)

# squeeze/unsqueeze - 删除/添加维度
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()         # (3, 4)
y = x.squeeze(0)        # (3, 1, 4)
y = x.unsqueeze(1)      # (1, 1, 3, 1, 4)

# flatten - 展平
y = x.flatten()         # 全部展平
y = x.flatten(1, 2)     # 从维度1到2展平
```

### 4.5 张量运算
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算
print(x + y)          # 加法
print(x - y)          # 减法
print(x * y)          # 乘法
print(x / y)          # 除法
print(x ** 2)         # 平方
print(torch.sqrt(x))  # 平方根

# 矩阵运算
a = torch.randn(2, 3)
b = torch.randn(3, 4)

print(torch.mm(a, b))      # 矩阵乘法
print(torch.matmul(a, b))  # 更通用的乘法
print(a @ b)               # @ 运算符

# 批量矩阵乘法
batch_a = torch.randn(10, 2, 3)
batch_b = torch.randn(10, 3, 4)
result = torch.bmm(batch_a, batch_b)  # (10, 2, 4)

# 点积
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(torch.dot(x, y))  # 32.0

# 聚合运算
x = torch.randn(3, 4)
print(torch.sum(x))         # 总和
print(torch.mean(x))        # 平均值
print(torch.max(x))         # 最大值
print(torch.min(x))         # 最小值
print(torch.argmax(x))      # 最大值索引
print(torch.argmin(x))      # 最小值索引

# 指定维度的聚合
print(x.sum(dim=0))         # 每列求和
print(x.mean(dim=1))        # 每行求平均
print(x.max(dim=0))         # 返回(values, indices)
```

### 4.6 张量拼接和分割
```python
# 拼接
x = torch.randn(2, 3)
y = torch.randn(2, 3)

z = torch.cat([x, y], dim=0)  # (4, 3) 沿行拼接
z = torch.cat([x, y], dim=1)  # (2, 6) 沿列拼接

z = torch.stack([x, y], dim=0)  # (2, 2, 3) 新增维度

# 分割
x = torch.randn(6, 4)
chunks = torch.chunk(x, 3, dim=0)  # 分成3块
splits = torch.split(x, 2, dim=0)   # 每块大小为2
```

### 4.7 广播机制
```python
# PyTorch自动扩展维度进行运算
x = torch.randn(3, 1)
y = torch.randn(1, 4)
z = x + y  # (3, 4)

# 标量广播
x = torch.randn(2, 3)
y = x + 10  # 10自动广播到所有元素
```

---

## 5. 自动微分机制

### 5.1 requires_grad属性
```python
import torch

# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 计算
z = x ** 2 + y ** 3
print(z)  # tensor([31.], grad_fn=<AddBackward0>)

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # tensor([4.]) = 2*x
print(y.grad)  # tensor([27.]) = 3*y^2
```

### 5.2 计算图
```python
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# 构建计算图
a = x + y
b = a * x
c = b.mean()

# 反向传播
c.backward()

print(x.grad)  # dc/dx
print(y.grad)  # dc/dy
```

### 5.3 梯度清零
```python
x = torch.tensor([1.0], requires_grad=True)

# 第一次计算
y = x ** 2
y.backward()
print(x.grad)  # tensor([2.])

# 第二次计算(梯度累积)
y = x ** 3
y.backward()
print(x.grad)  # tensor([5.]) = 2 + 3

# 清零梯度
x.grad.zero_()
y = x ** 4
y.backward()
print(x.grad)  # tensor([4.])
```

### 5.4 detach和no_grad
```python
# detach - 分离计算图
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y.detach()  # z不再跟踪梯度
w = z * 3
# w.backward()  # 报错,因为z已经分离

# no_grad - 临时禁用梯度
x = torch.randn(3, 3, requires_grad=True)
with torch.no_grad():
    y = x * 2  # y不需要梯度
    print(y.requires_grad)  # False

# 推理时常用
@torch.no_grad()
def predict(model, x):
    return model(x)
```

### 5.5 高阶导数
```python
# 计算二阶导数
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 一阶导数
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(dy_dx)  # tensor([12.]) = 3*x^2

# 二阶导数
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # tensor([12.]) = 6*x
```

### 5.6 自定义反向传播
```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

# 使用
my_func = MyFunction.apply
x = torch.tensor([2.0], requires_grad=True)
y = my_func(x)
y.backward()
print(x.grad)  # tensor([4.])
```

---

## 6. 神经网络基础

### 6.1 nn.Module基础
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化
model = SimpleNet(784, 128, 10)
print(model)

# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 6.2 常用层
```python
# 全连接层
fc = nn.Linear(in_features=100, out_features=50)

# 卷积层
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
conv1d = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=5)

# 池化层
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2)

# Dropout
dropout = nn.Dropout(p=0.5)

# BatchNorm
bn1d = nn.BatchNorm1d(num_features=100)
bn2d = nn.BatchNorm2d(num_features=64)

# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
```

### 6.3 Sequential容器
```python
# 使用Sequential快速构建网络
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
)

# 使用OrderedDict命名层
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.5)),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(0.5)),
    ('fc3', nn.Linear(128, 10))
]))
```

### 6.4 ModuleList和ModuleDict
```python
# ModuleList - 列表式管理
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(5)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ModuleDict - 字典式管理
class MyModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'linear1': nn.Linear(10, 20),
            'linear2': nn.Linear(20, 10)
        })
    
    def forward(self, x):
        x = self.layers['linear1'](x)
        x = self.layers['linear2'](x)
        return x
```

### 6.5 参数初始化
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier初始化
        nn.init.xavier_uniform_(m.weight)
        # 常数初始化偏置
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # Kaiming初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model = SimpleNet(784, 128, 10)
model.apply(init_weights)

# 手动初始化
nn.init.normal_(model.fc1.weight, mean=0, std=0.01)
nn.init.zeros_(model.fc1.bias)
```

---

## 7. 数据处理

### 7.1 Dataset类
```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# 创建数据集
data = np.random.randn(1000, 10).astype(np.float32)
labels = np.random.randint(0, 2, 1000).astype(np.int64)
dataset = CustomDataset(data, labels)

# 访问数据
x, y = dataset[0]
print(f"样本: {x.shape}, 标签: {y}")
```

### 7.2 DataLoader
```python
# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # GPU训练时加速
)

# 迭代数据
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
    if batch_idx >= 2:
        break
```

### 7.3 图像数据增强
```python
from torchvision import transforms
from PIL import Image

# 定义变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 应用变换
# img = Image.open('image.jpg')
# img_transformed = transform(img)
```

### 7.4 内置数据集
```python
from torchvision import datasets

# MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# CIFAR-10
cifar_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# ImageNet
# imagenet_dataset = datasets.ImageNet(
#     root='./data',
#     split='train',
#     transform=transform
# )
```

### 7.5 自定义图像数据集
```python
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 假设目录结构: root_dir/class_name/image.jpg
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

---

## 8. 训练神经网络

### 8.1 完整训练流程
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # 打印epoch信息
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
```

### 8.2 验证和测试
```python
def evaluate(model, data_loader, criterion):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for data, target in data_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 使用
# val_loss, val_acc = evaluate(model, val_loader, criterion)
# print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
```

### 8.3 学习率调度
```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# StepLR - 每N个epoch降低学习率
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# ReduceLROnPlateau - 当指标停止改善时降低
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# CosineAnnealingLR - 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# 在训练循环中使用
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    
    # 更新学习率
    scheduler.step()  # StepLR, CosineAnnealingLR
    # scheduler.step(val_loss)  # ReduceLROnPlateau
```

### 8.4 保存和加载模型
```python
# 保存整个模型
torch.save(model, 'model.pth')

# 只保存参数(推荐)
torch.save(model.state_dict(), 'model_weights.pth')

# 保存完整训练状态
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载模型
model = Net()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 加载完整训练状态
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 8.5 早停(Early Stopping)
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'best_model.pth')
        self.val_loss_min = val_loss

# 使用
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

### 8.6 梯度裁剪
```python
# 防止梯度爆炸
max_grad_norm = 1.0

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
```

### 8.7 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        
        # 自动混合精度
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
```

---

## 9. 卷积神经网络

### 9.1 卷积层详解
```python
import torch
import torch.nn as nn

# Conv2d参数
# in_channels: 输入通道数
# out_channels: 输出通道数(卷积核数量)
# kernel_size: 卷积核大小
# stride: 步长
# padding: 填充
# dilation: 膨胀

conv = nn.Conv2d(
    in_channels=3, 
    out_channels=64, 
    kernel_size=3, 
    stride=1, 
    padding=1
)

# 输入: (batch_size, channels, height, width)
x = torch.randn(8, 3, 224, 224)
output = conv(x)
print(output.shape)  # torch.Size([8, 64, 224, 224])

# 计算输出尺寸公式:
# output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

### 9.2 池化层
```python
# 最大池化
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 64, 56, 56)
output = maxpool(x)
print(output.shape)  # torch.Size([8, 64, 28, 28])

# 平均池化
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# 自适应池化(输出固定大小)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
x = torch.randn(8, 512, 7, 7)
output = adaptive_pool(x)
print(output.shape)  # torch.Size([8, 512, 1, 1])
```

### 9.3 经典CNN架构 - LeNet
```python
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 输入: (N, 1, 28, 28)
        x = self.relu(self.conv1(x))  # (N, 6, 24, 24)
        x = self.pool(x)               # (N, 6, 12, 12)
        x = self.relu(self.conv2(x))  # (N, 16, 8, 8)
        x = self.pool(x)               # (N, 16, 4, 4)
        x = x.view(x.size(0), -1)     # (N, 256)
        x = self.relu(self.fc1(x))    # (N, 120)
        x = self.relu(self.fc2(x))    # (N, 84)
        x = self.fc3(x)               # (N, 10)
        return x
```

### 9.4 VGG网络
```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### 9.5 ResNet残差网络
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### 9.6 迁移学习
```python
from torchvision import models

# 加载预训练模型
resnet = models.resnet50(pretrained=True)

# 冻结所有参数
for param in resnet.parameters():
    param.requires_grad = False

# 替换最后的全连接层
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10个类别

# 只训练最后一层
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# 或者微调整个网络
for param in resnet.parameters():
    param.requires_grad = True

# 使用不同学习率
optimizer = optim.Adam([
    {'params': resnet.conv1.parameters(), 'lr': 1e-5},
    {'params': resnet.layer1.parameters(), 'lr': 1e-5},
    {'params': resnet.layer2.parameters(), 'lr': 1e-4},
    {'params': resnet.layer3.parameters(), 'lr': 1e-4},
    {'params': resnet.layer4.parameters(), 'lr': 1e-3},
    {'params': resnet.fc.parameters(), 'lr': 1e-2}
])
```

---

## 10. 循环神经网络

### 10.1 RNN基础
```python
import torch
import torch.nn as nn

# 简单RNN
rnn = nn.RNN(
    input_size=10,    # 输入特征维度
    hidden_size=20,   # 隐藏层维度
    num_layers=2,     # RNN层数
    batch_first=True  # 输入格式(batch, seq, feature)
)

# 输入: (batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 10)
h0 = torch.zeros(2, 32, 20)  # (num_layers, batch, hidden_size)

output, hn = rnn(x, h0)
print(output.shape)  # (32, 50, 20)
print(hn.shape)      # (2, 32, 20)
```

### 10.2 LSTM
```python
# LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.5  # 层间dropout
)

x = torch.randn(32, 50, 10)
h0 = torch.zeros(2, 32, 20)
c0 = torch.zeros(2, 32, 20)  # cell state

output, (hn, cn) = lstm(x, (h0, c0))
print(output.shape)  # (32, 50, 20)
print(hn.shape)      # (2, 32, 20)
print(cn.shape)      # (2, 32, 20)
```

### 10.3 GRU
```python
# GRU
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(32, 50, 10)
h0 = torch.zeros(2, 32, 20)

output, hn = gru(x, h0)
```

### 10.4 文本分类模型
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, 
                           num_layers=2, batch_first=True, dropout=0.5)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        last_hidden = hn[-1]  # (batch, hidden_size)
        
        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, num_classes)
        return out

# 使用
model = TextClassifier(vocab_size=10000, embed_size=100, 
                      hidden_size=256, num_classes=5)
```

### 10.5 序列到序列模型(Seq2Seq)
```python
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        hidden, cell = self.encoder(source)
        
        x = target[:, 0].unsqueeze(1)  # <SOS> token
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t:t+1] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            x = target[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
```

### 10.6 双向LSTM
```python
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # bidirectional=True
        self.lstm = nn.LSTM(embed_size, hidden_size, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True, dropout=0.5)
        
        # 双向所以是 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # 拼接前向和后向的最后隐藏状态
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        out = self.fc(hidden)
        return out
```

---

## 11. Transformer架构

### 11.1 注意力机制
```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask(可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

### 11.2 多头注意力
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # x: (batch, num_heads, seq_len, d_k)
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并头
        output = self.combine_heads(attn_output)
        
        # 最后的线性层
        output = self.W_o(output)
        
        return output, attn_weights
```

### 11.3 位置编码
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
```

### 11.4 Transformer编码器层
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 11.5 完整Transformer模型
```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# 使用示例
model = TransformerEncoder(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_len=512,
    dropout=0.1
)
```

---

## 12. 高级技巧

### 12.1 自定义损失函数
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### 12.2 标签平滑
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

### 12.3 权重衰减和正则化
```python
# L2正则化(权重衰减)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Dropout
dropout = nn.Dropout(p=0.5)

# DropConnect
class DropConnect(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.size()) > self.p).float()
            return x * mask / (1 - self.p)
        return x
```

### 12.4 批归一化和层归一化
```python
# Batch Normalization
bn1d = nn.BatchNorm1d(num_features=128)
bn2d = nn.BatchNorm2d(num_features=64)

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=512)

# Group Normalization
gn = nn.GroupNorm(num_groups=32, num_channels=128)

# Instance Normalization
in2d = nn.InstanceNorm2d(num_features=64)
# Instance Normalization
in2d = nn.InstanceNorm2d(num_features=64)
```
# Instance Normalization
in2d = nn.InstanceNorm2d(num_features=64)
```

### 12.5 学习率预热(Warmup)
```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, d_model):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 使用
optimizer = optim.Adam(model.parameters(), lr=1)
scheduler = WarmupScheduler(optimizer, warmup_steps=4000, d_model=512)
```

### 12.6 梯度累积
```python
# 当GPU内存不足时，可以通过梯度累积模拟大batch
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 归一化损失
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 12.7 模型集成
```python
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # 平均预测
        return torch.mean(torch.stack(outputs), dim=0)

# 使用
model1 = ResNet18()
model2 = VGG16()
model3 = DenseNet()
ensemble = EnsembleModel([model1, model2, model3])
```

### 12.8 知识蒸馏
```python
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    # 软目标损失
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    
    # 硬目标损失
    hard_loss = nn.functional.cross_entropy(student_logits, true_labels)
    
    # 组合损失
    return alpha * soft_loss + (1 - alpha) * hard_loss

# 训练循环
teacher_model.eval()
student_model.train()

for data, target in train_loader:
    with torch.no_grad():
        teacher_logits = teacher_model(data)
    
    student_logits = student_model(data)
    loss = distillation_loss(student_logits, teacher_logits, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 12.9 对抗训练
```python
def fgsm_attack(model, data, target, epsilon=0.3):
    """快速梯度符号法(FGSM)"""
    data.requires_grad = True
    
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    
    model.zero_grad()
    loss.backward()
    
    # 生成对抗样本
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

# 对抗训练
for data, target in train_loader:
    # 正常训练
    output = model(data)
    loss = criterion(output, target)
    
    # 对抗训练
    adv_data = fgsm_attack(model, data, target, epsilon=0.1)
    adv_output = model(adv_data)
    adv_loss = criterion(adv_output, target)
    
    # 总损失
    total_loss = loss + 0.5 * adv_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### 12.10 混合训练(Mixup)
```python
def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 使用
for data, target in train_loader:
    mixed_data, target_a, target_b, lam = mixup_data(data, target, alpha=1.0)
    
    output = model(mixed_data)
    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 13. 项目实战

### 13.1 图像分类完整项目
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# 1. 数据准备
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, 
                               download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 2. 模型定义
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# 3. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 4. 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 5. 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 6. 训练循环
num_epochs = 50
best_acc = 0.0
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with accuracy: {best_acc:.2f}%')

# 7. 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
```

### 13.2 文本情感分析
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 1. 数据预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 文本转索引
        tokens = text.split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # 填充或截断
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)

# 2. 构建词汇表
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

# 3. 模型定义
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 连接前向和后向的最后隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return output

# 4. 训练
# 假设已有texts和labels
# vocab = build_vocab(texts)
# dataset = TextDataset(texts, labels, vocab)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = SentimentClassifier(vocab_size=len(vocab), embed_dim=100, 
#                            hidden_dim=256, num_classes=2)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环与图像分类类似
```

### 13.3 目标检测(使用预训练模型)
```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# 1. 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. 预处理图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image)
    return image, image_tensor

# 3. 预测
def detect_objects(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    # 过滤低置信度检测
    keep = predictions['scores'] > threshold
    boxes = predictions['boxes'][keep]
    labels = predictions['labels'][keep]
    scores = predictions['scores'][keep]
    
    return boxes, labels, scores

# 4. 可视化
def visualize_predictions(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        text = f'Class: {label}, Score: {score:.2f}'
        draw.text((x1, y1-10), text, fill='red')
    
    return image

# 使用
# image, image_tensor = preprocess_image('image.jpg')
# boxes, labels, scores = detect_objects(model, image_tensor)
# result_image = visualize_predictions(image, boxes, labels, scores)
# result_image.show()
```

### 13.4 生成对抗网络(GAN)
```python
# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# 判别器
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# GAN训练
import numpy as np

latent_dim = 100
img_shape = (1, 28, 28)

generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练循环
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # 真实和假标签
        valid = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)
        
        real_imgs = imgs
        
        # ---------------------
        #  训练生成器
        # ---------------------
        optimizer_G.zero_grad()
        
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)
        
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        if i % 100 == 0:
            print(f'[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] '
                  f'[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')
```

---

## 14. 性能优化

### 14.1 数据加载优化
```python
# 使用多进程加载
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # 多进程
    pin_memory=True,      # 锁页内存
    prefetch_factor=2,    # 预取因子
    persistent_workers=True  # 保持worker进程
)

# 自定义collate_fn加速
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    imgs = torch.stack(imgs, 0)
    return imgs, targets

train_loader = DataLoader(dataset, collate_fn=fast_collate)
```

### 14.2 模型并行
```python
# DataParallel - 单机多卡
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# DistributedDataParallel - 分布式训练(更高效)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    model = YourModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 训练代码
    
    cleanup()

# 使用
import torch.multiprocessing as mp
world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### 14.3 梯度检查点
```python
# 节省内存的技术
from torch.utils.checkpoint import checkpoint

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
    
    def forward(self, x):
        # 使用checkpoint减少内存
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

### 14.4 模型量化
```python
# 动态量化
import torch.quantization

model_fp32 = YourModel()
model_fp32.eval()

# 动态量化
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},  # 要量化的层类型
    dtype=torch.qint8
)

# 静态量化
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# 校准
with torch.no_grad():
    for data, _ in calibration_loader:
        model_fp32_prepared(data)

model_int8 = torch.quantization.convert(model_fp32_prepared)
```

### 14.5 模型剪枝
```python
import torch.nn.utils.prune as prune

# 非结构化剪枝
model = YourModel()
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)

# L1非结构化剪枝
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,  # 剪枝20%的参数
)

# 查看剪枝效果
print(f"Conv1 sparsity: {100. * float(torch.sum(model.conv1.weight == 0)) / float(model.conv1.weight.nelement()):.2f}%")

# 使剪枝永久化
for module, name in parameters_to_prune:
    prune.remove(module, name)
```

### 14.6 编译优化
```python
# TorchScript优化
model = YourModel()
model.eval()

# 方法1: Tracing
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 方法2: Scripting
scripted_model = torch.jit.script(model)

# 保存
traced_model.save('model_traced.pt')

# 加载
loaded_model = torch.jit.load('model_traced.pt')

# torch.compile (PyTorch 2.0+)
compiled_model = torch.compile(model)
```

---

## 15. 部署与生产

### 15.1 ONNX导出
```python
import torch.onnx

model = YourModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 验证ONNX模型
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# 使用ONNX Runtime推理
import onnxruntime as ort
ort_session = ort.InferenceSession("model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outputs = ort_session.run(None, ort_inputs)
```

### 15.2 Flask API部署
```python
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# 加载模型
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    # 预处理
    img_tensor = transform(img).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return jsonify({
        'prediction': int(predicted.item()),
        'confidence': float(torch.max(torch.softmax(output, 1)).item())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 15.3 TorchServe部署
```bash
# 安装TorchServe
pip install torchserve torch-model-archiver

# 创建MAR文件
torch-model-archiver --model-name my_model \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pth \
    --handler image_classifier \
    --extra-files index_to_name.json

# 启动服务
torchserve --start --model-store model_store --models my_model=my_model.mar

# 推理请求
curl -X POST http://localhost:8080/predictions/my_model -T image.jpg
```

### 15.4 移动端部署(PyTorch Mobile)
```python
# 准备移动端模型
model = YourModel()
model.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)

# 优化移动端
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_model = optimize_for_mobile(traced_script_module)

# 保存
optimized_model._save_for_lite_interpreter("model_mobile.ptl")

# Android/iOS使用
# 在应用中加载.ptl文件并进行推理
```

### 15.5 监控和日志
```python
import logging
from torch.utils.tensorboard import SummaryWriter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard
writer = SummaryWriter('runs/experiment1')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    # 记录标量
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 记录学习率
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    
    # 记录模型参数分布
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    
    logger.info(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')

writer.close()

# 启动TensorBoard: tensorboard --logdir=runs
```

---

## 附录: 常见问题和解决方案

### A.1 GPU内存不足
```python
# 1. 减小batch size
# 2. 使用梯度累积
# 3. 使用混合精度训练
# 4. 使用梯度检查点
# 5. 清理缓存

# 清理GPU缓存
torch.cuda.empty_cache()

# 监控GPU内存使用
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
```

### A.2 训练速度慢
```python
# 1. 使用DataLoader的num_workers
# 2. 启用pin_memory
# 3. 使用GPU
# 4. 使用混合精度训练
# 5. 优化数据预处理

# 性能分析
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### A.3 模型不收敛
```python
# 1. 检查学习率
# 2. 检查数据归一化
# 3. 检查梯度
# 4. 使用梯度裁剪
# 5. 尝试不同的优化器

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item()}")
    else:
        print(f"{name}: No gradient")

# 可视化梯度
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
```

### A.4 过拟合问题
```python
# 1. 增加Dropout
# 2. 使用数据增强
# 3. 减小模型复杂度
# 4. 使用L2正则化
# 5. 早停

# 更强的正则化
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 增加dropout率
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 增加weight decay
```

### A.5 批归一化问题
```python
# 训练时记得设置model.train()
model.train()
output = model(input)

# 推理时记得设置model.eval()
model.eval()
with torch.no_grad():
    output = model(input)

# 冻结BatchNorm层
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
```

### A.6 多GPU训练问题
```python
# 保存DataParallel模型
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), 'model.pth')
else:
    torch.save(model.state_dict(), 'model.pth')

# 加载到单GPU
model = YourModel()
state_dict = torch.load('model.pth')
# 移除'module.'前缀
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
```

### A.7 学习资源推荐

**官方文档:**
- PyTorch官方教程: https://pytorch.org/tutorials/
- PyTorch文档: https://pytorch.org/docs/
- PyTorch论坛: https://discuss.pytorch.org/

**优秀课程:**
- CS231n: Convolutional Neural Networks for Visual Recognition
- CS224n: Natural Language Processing with Deep Learning
- Fast.ai Practical Deep Learning for Coders
- Deep Learning Specialization (Coursera)

**推荐书籍:**
- 《深度学习》(花书) - Ian Goodfellow
- 《动手学深度学习》- 李沐
- 《PyTorch深度学习实战》
- 《Deep Learning with PyTorch》

**GitHub资源:**
- PyTorch官方示例: https://github.com/pytorch/examples
- PyTorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning
- Transformers: https://github.com/huggingface/transformers

**论文和博客:**
- arXiv.org - 最新研究论文
- Papers with Code - 论文+代码实现
- Distill.pub - 可视化深度学习
- Jay Alammar的博客 - Transformer等架构可视化

---

## 总结

这份教程覆盖了PyTorch从基础到高级的完整内容:

### 基础部分 (第1-5章)
- 环境搭建和Python基础
- PyTorch核心概念
- 张量操作和自动微分
- 神经网络基础构建

### 进阶部分 (第6-8章)
- nn.Module深入使用
- 数据处理和增强
- 完整的训练流程
- 模型保存和加载

### 架构部分 (第9-11章)
- 卷积神经网络 (CNN)
- 循环神经网络 (RNN/LSTM/GRU)
- Transformer架构
- 经典模型实现

### 高级部分 (第12-15章)
- 高级训练技巧
- 实战项目案例
- 性能优化方法
- 模型部署方案

### 学习建议:

1. **循序渐进**: 按顺序学习,每个概念都要动手实践
2. **多做项目**: 通过实际项目巩固知识
3. **阅读源码**: 看PyTorch和优秀开源项目的源码
4. **关注前沿**: 跟进最新的研究和技术
5. **社区参与**: 在论坛、GitHub上交流学习

祝学习顺利! 🚀# PyTorch从零基础到精通完整教程