import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class Net(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc = nn.Linear(5*5*40, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 5*5*40)
        x = self.fc(x)
        return x

class NetQuant(nn.Module):

    def __init__(self, num_channels=1):
        super(NetQuant, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5*5*40, 10)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 5*5*40)
        x = self.fc(x)
        x = self.dequant(x)
        return x


# model = Net()  
# qconfig = get_default_qconfig("fbgemm")
# qconfig_dict = {"": qconfig}
# model_prepared = prepare_fx(model, qconfig_dict)
# post_training_quantize(model_prepared, train_loader)      # 这一步是做后训练量化
# model_int8 = convert_fx(model_prepared)

