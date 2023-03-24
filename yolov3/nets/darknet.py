import math
from collections import OrderedDict

import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # 先用一个1 x 1 的卷积 下降通道数(因为：inplanes其实就是planes[1]), 128-->64
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)

        # 再用一个3 x 3 的卷积 扩张通道数 64 --> 128  通过这种办法减少参数量
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x #残差边

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out) #卷积标准化
        out = self.relu2(out) #激活函数

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self, layers): # 先定义每一层 layers = [1, 2, 8, 8, 4]
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32 先对输入进来的图片压缩其宽和高
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = nn.LeakyReLU(0.1)

        #_make_layer进行残差块的堆叠
        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):# blocks限定了残差块堆叠的次数
        layers = []
        # 下采样，步长为2，卷积核大小为3(压缩图片，图片宽高减小，通道数增加)
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 加入残差结构
        self.inplanes = planes[1] #每次都将当前这次卷积之后的输出通道数，当做下次卷积的输入通道数
        # blocks限定了残差块堆叠的次数
        for i in range(0, blocks): #BasicBlock用来实现残差网络的结构
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x) #最初始的32通道的卷积
        x = self.bn1(x) #卷积初始化
        x = self.relu1(x) #激活函数

        x = self.layer1(x)
        x = self.layer2(x)
        # 因为yolov3会用到Darknet最后的三个特征层，所以这三个要加以区分
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
