import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)          # 输出层（不加激活函数，通常结合任务需要添加）
        return x

# 超参数
# input_size = 10   # 输入维度
# hidden_size = 32  # 隐藏层神经元数
# output_size = 2   # 输出维度（分类任务通常是类别数）

# # 实例化模型
# model = SimpleMLP(input_size, hidden_size, output_size)
# print(model)

# # 测试输入
# test_input = torch.randn(5, input_size)  # Batch size = 5, 输入维度 = 10
# test_output = model(test_input)
# print("Test Output:", test_output)
