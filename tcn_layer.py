import torch
import torch.nn as nn
#Applies weight normalization to a parameter in the given module.
from torch.nn.utils import weight_norm
import torch.nn.functional as F

# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Crop(nn.Module):
    def __init__(self, crop_size ):
        super(Crop, self).__init__()
        self.crop_size  = crop_size #这个chomp_size就是padding的值

    def forward(self, x):
        return x[:, :, :-self.crop_size ].contiguous()
    
# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalCasualLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalCasualLayer, self).__init__()
        # padding = (kernel_size - 1) * dilation
        padding = (kernel_size * dilation - dilation) // 2
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      stride,
            'padding':     padding,
            'dilation':    dilation
        }

        # -------------------------
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        # self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        # self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # --------------------------

        # # -------------------------
        # self.conv1 = nn.Conv1d(n_inputs, n_outputs, **conv_params)
        # self.bn1 = torch.nn.BatchNorm1d(n_outputs)
        # # self.crop1 = Crop(padding)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, **conv_params)
        # # self.crop2 = Crop(padding)
        # self.bn2 = torch.nn.BatchNorm1d(n_outputs)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)
        # # --------------------------
        # # -------------------------
        # self.conv1 = nn.Conv1d(n_inputs, n_outputs, **conv_params)
        # # self.crop1 = Crop(padding)
        # self.bn1 = F.normalize
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, **conv_params)
        # # self.crop2 = Crop(padding)
        # self.bn2 = F.normalize
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)
        # # --------------------------
 
 
        # self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1, self.conv2, self.crop2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1, self.conv2, self.bn2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1)
        #shortcut connect
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.bias is not None:
            self.bias.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)
    

#最后就是TCN的主网络了
class TemporalConvolutionNetwork(nn.Module):
 
    def __init__(self, num_inputs, num_channels, kernel_size = 2, dropout = 0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
 
    def forward(self, x):
        y = self.tcn(x)#[N,C_out,L_out=L_in]
        # return self.linear(y[:, :, -1])
        return self.linear(y[:, :, :]).squeeze()
    