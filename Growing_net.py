import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 定义动态层
class DynamicLayer(nn.Module):
    def __init__(self, input_features, output_features, init_weights=None, init_bias=None):
        super(DynamicLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        if init_weights is not None:
            self.weights = nn.Parameter(init_weights)
        else:
            self.weights = nn.Parameter(torch.randn(output_features, input_features))

        if init_bias is not None:
            self.bias = nn.Parameter(init_bias)
        else:
            self.bias = nn.Parameter(torch.randn(output_features))

        # 初始化损失记录
        self.running_loss = 0
        self.loss_count = 0

    def grow_neurons(self, new_output_features):
        additional_weights = nn.Parameter(torch.zeros(new_output_features, self.input_features))
        additional_bias = nn.Parameter(torch.zeros(new_output_features))
        self.weights = nn.Parameter(torch.cat([self.weights, additional_weights], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias, additional_bias]))
        self.output_features += new_output_features

    def adjust_input_dim(self, new_input_features):
        additional_weights = nn.Parameter(torch.zeros(self.output_features, new_input_features - self.input_features))
        self.weights = nn.Parameter(torch.cat([self.weights, additional_weights], dim=1))
        self.input_features = new_input_features

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def record_loss(self, loss):
        self.running_loss += loss
        self.loss_count += 1

    def get_average_loss(self):
        if self.loss_count == 0:
            return 0
        return self.running_loss / self.loss_count

    def reset_loss(self):
        self.running_loss = 0
        self.loss_count = 0

# 定义灵活的神经网络
class FlexibleNN(nn.Module):
    def __init__(self):
        super(FlexibleNN, self).__init__()
        # 创建一个包含所有层的数组
        self.layers = nn.ModuleList([
            DynamicLayer(784, 128),
            DynamicLayer(128, 64),
            DynamicLayer(64, 32),
            DynamicLayer(32, 10)
        ])

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平图像
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x

    def grow_network_if_needed(self, growth_threshold=3):
        # 检查每层是否需要增长
        for i in range(len(self.layers) - 1):  # 不包括最后一层
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            if layer.get_average_loss() > growth_threshold:
                old_shape = layer.weights.shape
                layer.grow_neurons(10)  # 增加10个神经元
                next_layer.adjust_input_dim(layer.output_features)  # 更新下一层的输入维度
                new_shape = layer.weights.shape
                print(f"Layer {i + 1} grown from {old_shape} to {new_shape}")
                layer.reset_loss()


# 准备数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建网络和优化器
net = FlexibleNN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 训练网络
for epoch in range(10):  # 遍历数据集两次
    net.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 更新每一层的损失记录
        for layer in net.layers[:-1]:  # 不包括最后一层
            layer.record_loss(loss.item())

        if i % 100 == 99:
            net.grow_network_if_needed()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# 测试网络性能
net.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'在测试集上的准确率: {100 * correct / total}%')

print('完成训练')
