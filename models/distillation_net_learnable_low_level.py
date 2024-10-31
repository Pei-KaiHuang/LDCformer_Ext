import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import torchvision.models as models


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class FM_Res18_learnable(nn.Module):
    def __init__(self):
        super(FM_Res18_learnable, self).__init__()
        from models.learnable_R18 import resnet18
        model_resnet = resnet18()
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        model_resnet.load_state_dict(state_dict, strict=False)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.base_mask = nn.Parameter(torch.ones([3,3])*0.11, requires_grad=False)
        self.pool_mask = nn.Parameter(torch.Tensor([[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1]]), requires_grad=True)

    def forward(self, input):
        theta = 0.1
        avg_mask = self.base_mask.expand(64,64,-1,-1)
        ldp_mask = self.pool_mask.expand(64,64,-1,-1)

        feature = self.conv1(input) 
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        # re_feature1 = F.adaptive_avg_pool2d(feature1, 32)
        re_feature1 = F.conv2d(feature1, weight = (1-theta)*avg_mask + theta*ldp_mask, stride=2, padding=1) # new added: LDP
        re_feature2 = F.adaptive_avg_pool2d(feature2, 32)
        re_feature3 = F.adaptive_avg_pool2d(feature3, 32)
        catfeat = torch.cat([re_feature1, re_feature2, re_feature3], 1)

        feature = self.layer4(feature3)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(feature, feature_norm)

        return feature, catfeat


class FM_Res18(nn.Module):
    def __init__(self):
        super(FM_Res18, self).__init__()
        from models.vanilla_R18 import resnet18
        model_resnet = resnet18()
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        model_resnet.load_state_dict(state_dict, strict=False)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        re_feature1 = F.adaptive_avg_pool2d(feature1, 32)
        re_feature2 = F.adaptive_avg_pool2d(feature2, 32)
        re_feature3 = F.adaptive_avg_pool2d(feature3, 32)
        catfeat = torch.cat([re_feature1, re_feature2, re_feature3], 1)

        feature = self.layer4(feature3)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature) 
        feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(feature, feature_norm)

        return feature, catfeat


class FE_Res18(nn.Module):
    def __init__(self):
        super(FE_Res18, self).__init__()
        model_resnet = models.resnet18(pretrained=True)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out


class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net.forward(self.grl_layer.forward(feature))
        return adversarial_out
    

class DG_model_AlAS_learnable(nn.Module):
    def __init__(self):
        super(DG_model_AlAS_learnable, self).__init__()
        self.backbone = FM_Res18_learnable() # Feature_Extractor_ResNet18_learnable 
        self.LiveEstor = Estmator_learnable() # Estmator_learnable  Estmator
        self.SpoofEstor = Estmator_learnable()# Estmator_learnable
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature, catfeat = self.backbone(input) 
        live_Pre = self.LiveEstor(catfeat)
        spoof_Pre = self.SpoofEstor(catfeat)
        classifier_out = self.classifier(feature, norm_flag)
        
        return classifier_out, live_Pre, spoof_Pre, feature 


class DG_model_AlAS_learnable_visualization(nn.Module):
    def __init__(self):
        super(DG_model_AlAS_learnable_visualization, self).__init__()
        self.backbone = FM_Res18_learnable_visualization()  # Feature_Extractor_ResNet18_learnable 
        self.LiveEstor = Estmator_learnable()  # Estmator_learnable  Estmator
        self.SpoofEstor = Estmator_learnable()  # Estmator_learnable
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature = self.backbone(input)
        return feature


class FM_Res18_learnable_visualization(nn.Module):
    def __init__(self):
        super(FM_Res18_learnable_visualization, self).__init__()
        from models.learnable_R18 import resnet18
        model_resnet = resnet18()
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        model_resnet.load_state_dict(state_dict, strict=False)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature) 
        feature = self.layer1[0].conv1(feature)
        # feature = self.layer1[0].bn1(feature)
        # feature = self.layer1[0].relu(feature)
        # feature = self.layer1[0](feature)
        # feature = self.layer1[0](feature)
        # feature1 = self.layer1(feature)
        # feature2 = self.layer2(feature1)
        # feature3 = self.layer3(feature2)
        return feature


class DG_model_AlAS(nn.Module):
    def __init__(self):
        super(DG_model_AlAS, self).__init__()
        self.backbone = FM_Res18() # Feature_Extractor_ResNet18_learnable 
        self.LiveEstor = Estmator_learnable() # Estmator_learnable  Estmator
        self.SpoofEstor = Estmator_learnable()# Estmator_learnable
        self.classifier = Classifier()

    def forward(self, input, norm_flag=True):
        feature, catfeat = self.backbone(input) 
        live_Pre = self.LiveEstor(catfeat)
        spoof_Pre = self.SpoofEstor(catfeat)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out
 


class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal


class conv3x3_learn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3_learn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        # CDC

        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff


class Estmator_learnable(nn.Module):
    def __init__(self, in_channels=448, out_channels=1, conv3x3=conv3x3_learn):  # conv3x3 384 960 conv3x3_learn
        super(Estmator_learnable, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Estmator(nn.Module):
    def __init__(self, in_channels=448, out_channels=1, conv3x3=conv3x3):  # conv3x3 384 960 conv3x3_learn
        super(Estmator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResNet_LS(nn.Module):
    def __init__(self, ):
        super(ResNet_LS, self).__init__()
        # #distangle_init_LS_with_ID
        self.FeatExtor_LS = models.resnet34(pretrained=True)
        self.FC_LS = torch.nn.Linear(1000, 2)

    def forward(self, x):
        if self.training:
            lsf = self.FeatExtor_LS(x)
            x_ls = self.FC_LS(lsf)
            return x_ls, lsf.detach()
        else:
            lsf = self.FeatExtor_LS(x)
            x_ls = self.FC_LS(lsf)
            return x_ls

class ResNet_ID(nn.Module):  
    def __init__(self, output=2):
        super(ResNet_ID, self).__init__()
        # #distangle_init_LS_with_ID
        self.FeatExtor_ID = models.resnet34(pretrained=True) 
        self.FC_ID = torch.nn.Linear(1000, output)
    def forward(self, x,type="image"):
        if self.training:

            idf = self.FeatExtor_ID(x)
            x_id = self.FC_ID(idf)
            return x_id 
        else:
            if type == "image":
                idf = self.FeatExtor_ID(x)
                x_id = self.FC_ID(idf)
                return x_id
            else:
                x_id = self.FC_ID(x)
                return x_id


class ResNet_ID_LS_invariant(nn.Module):
    def __init__(self, output=2):
        super(ResNet_ID_LS_invariant, self).__init__()
        self.FeatExtor_LS = models.resnet34(pretrained=True)
        self.FeatExtor_ID = models.resnet34(pretrained=True)
        self.FC_LS = torch.nn.Linear(1000, 2)
        self.FC_ID = torch.nn.Linear(1000, output)
        self.eval_type = "LS"  # ID LS

    def forward(self, x):
        if self.training:
            idf = self.FeatExtor_ID(x)
            lsf = self.FeatExtor_LS(x)

            x_id = self.FC_ID(idf)
            x_ls = self.FC_LS(lsf)

            x_id_invariant = self.FC_ID(lsf)
            return x_id, x_ls, x_id_invariant

        else:
            if self.eval_type == "ID":
                idf = self.FeatExtor_ID(x)
                x = self.FC_ID(idf)
                return x
            else:
                x = self.FeatExtor_LS(x)
                x = self.FC_LS(x)
                return x

###################### Intra  model ######################

class ResNet_Amodel(nn.Module):
    def __init__(self, ):
        super(ResNet_Amodel, self).__init__()
        # #distangle_init_LS_with_ID
        self.FeatExtor_LS = models.resnet18(pretrained=True)
        self.FC_LS = torch.nn.Linear(1000, 2)

    def forward(self, x): 
        x = self.FeatExtor_LS(x)
        x = self.FC_LS(x)
        return x

###################### Intra  model ######################

def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)
class Demo (nn.Module):  
    def __init__(self, output=2):
        super(Demo, self).__init__()
        self.num_features =3
        self.gamma = torch.nn.Parameter(torch.ones(1, self.num_features, 1, 1) * 0.3)
        self.beta = torch.nn.Parameter(torch.ones(1, self.num_features, 1, 1) * 0.5)
    def forward(self, x):  

        gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                 device=self.gamma.device) * softplus(self.gamma)).expand_as(x)
        beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
            self.beta)).expand_as(x)
        x = gamma * x + beta
        
        return x


if __name__ == '__main__':
    x = Variable(torch.ones(1, 3, 256, 256).cuda())
    model = DG_model_AlAS_learnable().cuda()
    model(x)