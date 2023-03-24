from collections import OrderedDict

import torch
import torch.nn as nn

from nets.attention import se_block
from nets.darknet import darknet53
from nets.attention_cbam import cbam_block
from nets.attention_eca import eca_block

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1), # 1 x 1 的卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3), # 3 x 3卷积进行特征提取
        conv2d(filters_list[1], filters_list[0], 1), #1 x 1 的卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3), # 3 x 3卷积进行特征提取
        conv2d(filters_list[1], filters_list[0], 1), #1 x 1 的卷积调整通道数,最终将通道数调整成512
        # 分类预测和回归预测
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75 (实际使用的只有行人一个类，所以 num_classes = 1)
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = len(anchors_mask[0]) * (num_classes + 5)
        #------------------------------------------------------------------------#   3(每个网格点上默认的三个先验框)  *  20 + 5
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1) #调整通道数
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')# 调整特征层的宽和高
        # 之后通过self.last_layer1_upsample上面的得到26 x 26 x 256 的特征层
        # 进行concat操作
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        # # SENet实现注意力机制
        # self.feat1_att = se_block(256)
        # self.feat2_att = se_block(512)
        # self.feat3_att = se_block(1024)

        # cbam实现注意力机制
        # self.feat1_att = cbam_block(256)
        # self.feat2_att = cbam_block(512)
        # self.feat3_att = cbam_block(1024)

        #eca实现注意力机制
        self.feat1_att = eca_block(256)
        self.feat2_att = eca_block(512)
        self.feat3_att = eca_block(1024)



    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        x2 = self.feat1_att(x2)
        x1 = self.feat2_att(x1)
        x0 = self.feat3_att(x0)


        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,75,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # 因为返回的是一个Sequential，而Sequential里面的forward函数需要x0，所以是对x0进行处理，让x0经过前五次的卷积操作
        out0_branch = self.last_layer0[:5](x0)# 保存5次卷积的结果
        out0        = self.last_layer0[5:](out0_branch)# 保存回归预测和分类预测的结果

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch) #返回一个Sequential,而Sequential里面的forward函数需要一个参数，所以传入out0_branch
        x1_in = self.last_layer1_upsample(x1_in) # 和上面同理

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,75,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第三个特征层
        #   out2 = (batch_size,75,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2

