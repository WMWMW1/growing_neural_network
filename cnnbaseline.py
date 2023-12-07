import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 input channels, 64 output channels, 3x3 kernel
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = x.view(-1, 64 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 准备数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建CNN模型和优化器
cnn_net = SimpleCNN()
cnn_optimizer = torch.optim.Adam(cnn_net.parameters(), lr=0.001)
cnn_loss_fn = nn.CrossEntropyLoss()

# 训练CNN模型
for epoch in range(10):
    cnn_net.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data

        cnn_optimizer.zero_grad()
        outputs = cnn_net(inputs)
        loss = cnn_loss_fn(outputs, labels)
        loss.backward()
        cnn_optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# 测试CNN模型性能
cnn_net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = cnn_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'CNN模型在测试集上的准确率: {100 * correct / total}%')

print('完成训练')
