import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from network_module import *
import utils

#-----------------------------------------------
# Content based Image Retrieval Network(CBIRNet)
#-----------------------------------------------

class Backbone(nn.Module):
    def __init__(self, opt, loadweights=True):
        super(Backbone, self).__init__()
        if opt.backbone_type == 'vgg16':
            vgg16 = models.vgg16(pretrained=loadweights)
            self.backbone = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), vgg16.features[1:30])
        elif opt.backbone_type == 'resnet50':
            resnet50 = models.resnet50(pretrained=loadweights)
            self.backbone = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), resnet50.bn1, resnet50.relu, resnet50.maxpool, 
                                          resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4)
        elif opt.backbone_type == 'resnet101':
            resnet101 = models.resnet101(pretrained=loadweights)
            self.backbone = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), resnet101.bn1, resnet101.relu, resnet101.maxpool, 
                                          resnet101.layer1, resnet101.layer2, resnet101.layer3, resnet101.layer4)
        elif opt.backbone_type == 'resnet152':
            resnet152 = models.resnet152(pretrained=loadweights)
            self.backbone = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), resnet152.bn1, resnet152.relu, resnet152.maxpool, 
                                          resnet152.layer1, resnet152.layer2, resnet152.layer3, resnet152.layer4)
        else:
            raise NotImplementedError('CBIRNet backbone_type %s not implemented' % opt.backbone_type)
    
    def forward(self, x):
        return self.backbone(x)
    
class CBIRNet(nn.Module):
    def __init__(self, opt):
        super(CBIRNet, self).__init__()
        self.opt = opt
        if opt.backbone_type == 'vgg16':
            start_channels = 256
        elif opt.backbone_type == 'resnet50':
            start_channels = 1024
        elif opt.backbone_type == 'resnet101':
            start_channels = 1024
        elif opt.backbone_type == 'resnet152':
            start_channels = 1024
        self.conv1 = Conv2dLayer(start_channels * 2, start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.conv2 = Conv2dLayer(start_channels, int(start_channels / 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(int(start_channels / 2), 128, bias=True)

    def forward(self, x, kcm): 
        kcm = F.interpolate(kcm, size=x.size()[2:], mode='bilinear', align_corners=True)
        # print(x.shape)
        # print(kcm.shape)       
        fused_features = self.opt.kcm_scale * x * kcm + (1 - self.opt.kcm_scale) * x
        fused_features = self.conv1(fused_features)
        fused_features = self.conv2(fused_features)
        # print(fused_features.shape)
        fused_features = self.global_pooling(fused_features)
        fused_features = fused_features.view(fused_features.size(0), -1)
        preds = self.fc_layer(fused_features)
        return preds

class ImgRetriever(nn.Module):
    def __init__(self, opt, loadweights=True):
        super(ImgRetriever, self).__init__()
        self.backbone = Backbone(opt, loadweights=loadweights)
        self.retrieval_module = CBIRNet(opt)

    def forward(self, x, kcm):
        fetures = self.backbone(x)
        preds = self.retrieval_module(fetures, kcm)
        return preds


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_type', type = str, default = 'resnet50', help = 'choose the backbone')
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--net_type', type = str, default = 'ImgRetriever', help = 'Used for composition classification')
    parser.add_argument('--load_com_path', type = str, default = './cc_model/ccnet_best.pth', help = 'the load path of com models')
    parser.add_argument('--load_path', type = str, default = None, help = 'the load path of models')
    parser.add_argument('--comp_types_num', type = int, default = 9, help = 'composition types num')
    parser.add_argument('--kcm_scale', type = int, default = 0.5, help = 'the scale of kcm')
    opt = parser.parse_args()

    print('backbone_type: %s' % opt.backbone_type)
    
    net = utils.create_net(opt)
    com_net = utils.create_net(opt, is_com = True)
    com_net.eval()
    
    net = net.cuda()
    com_net = com_net.cuda()

    a = torch.randn(1, 1, 256, 256).cuda()
    _, kcm = com_net(a)
    preds = net(a, kcm)
    print(preds.shape, kcm.shape)
    print(preds)
    
    file_path = "./dev_samples/preds.pt"
    torch.save(preds, file_path)
    loaded_preds = torch.load(file_path)
    print(loaded_preds.shape)
    print(loaded_preds)
    