import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import einops

from network_module import *

#-----------------------------------------------
#       Composition Clues Network(CCNet)
#-----------------------------------------------

class Backbone(nn.Module):
    def __init__(self, opt, loadweights=True):
        super(Backbone, self).__init__()
        if opt.backbone_type == 'vgg16':
            backbone = models.vgg16(pretrained=loadweights)
            self.feature1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), backbone.features[1:6])  # /2
            # self.feature1 = nn.Sequential(backbone.features[:6])    # /4
            self.feature2 = nn.Sequential(backbone.features[6:10])    # /4
            self.feature3 = nn.Sequential(backbone.features[10:17])   # /8
            self.feature4 = nn.Sequential(backbone.features[17:30])   # /16
        elif opt.backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=loadweights)
            self.feature1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
            self.feature2 = backbone.layer2
            self.feature3 = backbone.layer3
            self.feature4 = backbone.layer4
        elif opt.backbone_type == 'resnet101':
            backbone = models.resnet101(pretrained=loadweights)
            self.feature1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
            self.feature2 = backbone.layer2
            self.feature3 = backbone.layer3
            self.feature4 = backbone.layer4  
        elif opt.backbone_type == 'resnet152':
            backbone = models.resnet152(pretrained=loadweights)
            self.feature1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
            self.feature2 = backbone.layer2
            self.feature3 = backbone.layer3
            self.feature4 = backbone.layer4
        else:
            raise NotImplementedError('CCNet backbone_type %s not implemented' % opt.backbone_type)

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f2,f3,f4
    
class CCNet(nn.Module):
    def __init__(self, opt):
        super(CCNet, self).__init__()
        self.opt = opt
        if opt.backbone_type == 'vgg16':
            start_channels = 128
        elif opt.backbone_type == 'resnet50':
            start_channels = 512
        elif opt.backbone_type == 'resnet101':
            start_channels = 512
        elif opt.backbone_type == 'resnet152':
            start_channels = 512
        self.conv1 = Conv2dLayer(start_channels * 4, start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.conv2 = Conv2dLayer(start_channels * 2, start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.conv3 = Conv2dLayer(start_channels * 2, start_channels, 1, 1, 0, pad_type = opt.pad, activation = opt.activ)
        self.conv4 = Conv2dLayer(start_channels, start_channels, 1, 1, 0, pad_type = opt.pad, activation = opt.activ)
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Linear(start_channels, opt.comp_types_num, bias=True)

    def forward(self, f2, f3, f4):
        x = self.conv1(f4)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f3
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f2
        x = self.conv4(x)
        gap = self.GAP(x)
        preds = self.fc_layer(gap)
        conf   = F.softmax(preds, dim=1)
        with torch.no_grad():
            B,C,H,W = x.shape
            w  = self.fc_layer.weight.data # cls_num, channels
            trans_w = einops.repeat(w, 'n c -> b n c', b=B)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x) # b n hw
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=H, w=W)
            kcm = torch.sum(conf[:,:,None,None] * cam, dim=1, keepdim=True)
            kcm = F.interpolate(kcm, scale_factor=4, mode='bilinear', align_corners=True)
            return preds, kcm

class ComClassifier(nn.Module):
    def __init__(self, opt, loadweights=True):
        super(ComClassifier, self).__init__()
        self.backbone = Backbone(opt, loadweights=loadweights)
        self.composition_module = CCNet(opt)

    def forward(self, x):
        # print(self.backbone)
        f2,f3,f4 = self.backbone(x)
        preds,kcm = self.composition_module(f2,f3,f4)
        return preds,kcm


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_type', type = str, default = 'vgg16', help = 'choose the backbone')
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    opt = parser.parse_args()

    print('backbone_type: %s' % opt.backbone_type)
    net = ComClassifier(opt, loadweights=True)

    a = torch.randn(1, 1, 256, 256)
    preds, kcm = net(a)
    print(preds.shape, kcm.shape)
    print(preds)
    