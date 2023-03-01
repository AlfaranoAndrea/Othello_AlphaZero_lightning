import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Popular resnet like NN for board game
"""
class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride, padding=(1,1)),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.block= nn.Sequential(
            ConvBlock(inputs, outputs),
            nn.Conv2d(outputs, outputs, 3, stride, padding=(1,1)),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)+x

class Network(nn.Module):
    def __init__(self, inputs, outputs, num_of_res_layers, board_size, action_size):
        super().__init__()
        self.conv3x3 = ConvBlock(inputs, outputs, 1)
        res_blocks = [ResidualBlock(outputs,outputs, 1) for _ in range(num_of_res_layers)]
        self.res_layers = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(outputs, 2, 1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2 * board_size , action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(outputs, 1, 1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(board_size, outputs),
            nn.ReLU(),
            nn.Linear(outputs, 1)
        )
    def forward(self, x):
        out = self.conv3x3(x)
        out = self.res_layers(out)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)
        return F.log_softmax(policy_out,dim=1), torch.tanh(value_out)