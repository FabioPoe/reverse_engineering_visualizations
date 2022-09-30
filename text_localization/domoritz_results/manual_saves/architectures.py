import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#in this file i saved all network architectures, which seemed somewhat promising, but for the sake of keeping it simple i only kept the original

#domoritz original cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=[7, 7],stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3, 3],stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3, 3],stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=1, kernel_size=[2, 2],stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def ramp(self,x):
        #ramp activation fct: https://iopscience.iop.org/article/10.1088/1757-899X/224/1/012054/pdf
        x[x < -1] = -1
        x[x > 1] = 1
        return x
    
    def forward(self, x):
        x = self.ramp(self.conv1(x))
        x = self.ramp(self.conv2(x))
        x = self.ramp(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
