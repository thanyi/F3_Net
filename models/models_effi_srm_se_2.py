"""
这个文件是包含了efficientNet+srm
            FAD+SRM+SE
            同时尝试只使用一个efficientNet网络
"""
from PIL import Image
from torchvision import transforms

import utils.f3net_conf as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types

import timm.models.efficientnet as effnet

SRM_npy = np.load(r'E:\Desktop\DF_detection\F3_Net\models\SRM_Kernels.npy')


class SEblock(nn.Module):
    def __init__(self, in_chnls, ratio=16):
        super(SEblock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x * torch.sigmoid(out)



class EffNet(nn.Module):
    def __init__(self, arch='b7'):
        super(EffNet, self).__init__()
        fc_size = {'b1': 1280, 'b2': 1408, 'b3': 1536, 'b4': 1792,
                   'b5': 2048, 'b6': 2304, 'b7': 2560}
        assert arch in fc_size.keys()
        effnet_model = getattr(effnet, 'tf_efficientnet_%s_ns' % arch)
        self.encoder = effnet_model()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(fc_size[arch], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        # x = self.avg_pool(x).flatten(1)
        # x = self.dropout(x)
        # x = self.fc(x)
        return x


# Filter Module
class Filter(nn.Module):
    '''
    FAD模块的filter，添加进DCT变换之后的矩阵
        因为filter中设置grad为True所以是可以进行反向传播的
    '''

    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    '''
    FAD模块
        3个不同频率的filter和一个整体filter生成
        将DCT变换后的base filter分别和这四个点乘
        结果经cat后输出
    '''

    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 380, 380]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 380, 380]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 380, 380]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 380, 380]
        return out




class HPF_SRM(nn.Module):
    def __init__(self):
        super(HPF_SRM, self).__init__()
        self.SRM = torch.nn.Conv2d(1, 30, 5, 1, 0, bias=False)
        self.Weight = torch.from_numpy(SRM_npy)
        self.SRM.weight.data = self.Weight
        self.SRM.weight.requires_grad = False
        self.Padding = torch.nn.ReplicationPad2d(2)
    def forward(self, x):
        '''

        Args:
            x: 输入一个 [N,3,380,380]

        Returns: 一个cat拼接好了的返回变量

        '''
        srm_list = []
        for i in range(3):
            single_channel = x[:,i,:,:]
            single_channel = single_channel.unsqueeze(1)    # [N, 1, 380, 380]
            # print(single_channel.shape)
            y = self.SRM(self.Padding(single_channel))
            srm_list.append(y)

        out = torch.cat(srm_list, dim=1)
        # print(out.shape)
        return out


class F3Net(nn.Module):
    def __init__(self, num_classes=1, img_width=380, img_height=380, LFS_window_size=10, LFS_stride=2, LFS_M=6,
                 mode='Both',loss_mode="Logits", device=None):
        super(F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.fad_channel = 12
        self.srm_channel = 90
        self.se = SEblock(self.fad_channel+self.srm_channel)



        if mode == 'Both':
            self.FAD_head = FAD_Head(img_size)
            self.SRM_head = HPF_SRM()
            self.init_eff()

        # classifier
        self.relu = nn.ReLU(inplace=True)
        # effnet 的全连接

        if loss_mode == 'Logits':
            self.fc = nn.Linear(2560, 1)
        else:
            self.fc = nn.Linear(2560, 2)



        self.dp = nn.Dropout(p=0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # 指定输出的通道大小就是1*1



    def init_eff(self):
        '''
        默认加载EffNet
        Returns:

        '''
        self.eff = EffNet("b7")

        state_dict = get_eff_state_dict()

        conv1_data = state_dict['encoder.conv_stem.weight'].data

        self.eff.load_state_dict(state_dict, False)

        self.eff.encoder.conv_stem = nn.Conv2d(self.srm_channel + self.fad_channel, 64, 3, 2, 0, bias=False)




        for i in range(int((self.srm_channel+self.fad_channel) / 3)):
            self.eff.encoder.conv_stem.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float((self.srm_channel+self.fad_channel )/ 3)


    def forward(self, x):
        if self.mode == 'FAD':
            fea_FAD = self.FAD_head(x)
            fea_FAD = self.FAD_eff(fea_FAD)
            fea_FAD = self._norm_fea(fea_FAD)
            y = fea_FAD

        if self.mode == 'Original':
            fea = self.eff.features(x)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'Both':
            fea_FAD = self.FAD_head(x)
            # fea_FAD = self.FAD_eff(fea_FAD)
            # fea_FAD = self._norm_fea(fea_FAD)
            fea_SRM = self.SRM_head(x)
            # fea_SRM = self.SRM_eff(fea_SRM)
            # fea_SRM = self._norm_fea(fea_SRM)

            y = torch.cat((fea_FAD, fea_SRM), dim=1)
            #print("y:"+str(y.size()))
            y = self.se(y)
            y = self.eff(y)
            y = self._norm_fea(y)

        # print(y.shape)
        f = self.dp(y)
        f = self.fc(f)

        return f

    def _norm_fea(self, fea):
        '''

        Args:
            fea: 特征图

        Returns: 经过了池化过后的特征

        '''
        f = self.relu(fea)
        # print("relu")
        f = self.avg_pool(f).flatten(1)  # 全局平均池化 + 在第一维全部展开
        # print("avg")
        # f = f.view(f.size(0), -1)  # f.size() 和 f.shape 一样 其中每个参数就是相应维度的数值 这一行作用感觉和上面一行一样的
        # print("view")
        return f


# utils
def DCT_mat(size):
    '''
    DCT变换函数
    '''
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    '''
    进行归一化的 δ(x)
    '''
    return 2. * torch.sigmoid(x) - 1.


def get_eff_state_dict(pretrained_path=config.efficient_pretrained_path):
    '''
    这个函数进行
        预训练模型加载
        模型字典将带有”pointwise“的tensor扩维数
        将预训练模型中带有“fc“的key的value删除
    Args:
        pretrained_path: 最开始的预训练模型

    Returns:

    '''
    # load EffNent
    state_dict = torch.load(pretrained_path)

    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict


if __name__ == '__main__':


    tf = transforms.Compose([
        lambda x: Image.open(x).convert("RGB"),  # string path => image data
        # transforms.Resize(380),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    path = r"C:\Users\ethanyi\AppData\Roaming\JetBrains\PyCharm2021.1\scratches\1.png"
    img = tf (path)
    img = img.unsqueeze(0)
    print(img.shape)
    # srm = HPF_SRM()
    # out  = srm(img)
    # print(out.shape)






    net = F3Net()
    out = net(img)
    # print(net.buffers)

