import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16): # ratio=16是 输入的特征层的特征长条的缩放比例
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # nn.AdaptiveAvgPool2d(1)中的1表示：输出的特征层的高和宽为1
        self.fc = nn.Sequential( # 全连接层
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # **Code: b,c,h,w = x.size()
        b, c, _, _ = x.size()
        # b,c,h,w --> b,c,1,1
        avg = self.avg_pool(x).view(b, c)
        # **Code: avg = self.avg_pool(x).view([b, c])

        #总的维度的变化：b，c -> b, c//ratio -> b, c -> b, c, 1, 1
        y = self.fc(avg).view(b, c, 1, 1)
        return x * y