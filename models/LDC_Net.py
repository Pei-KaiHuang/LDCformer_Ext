import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
 
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Feature_Extractor_R18(nn.Module):
    def __init__(self):
        super(Feature_Extractor_R18, self).__init__()
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
 
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        # self.bn = nn.BatchNorm1d(512)
    def forward(self, input): 
        # input = self.bn(input)
        self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
        classifier_out = self.classifier_layer(input)
        return classifier_out

class LDC_Net_Cs(nn.Module):
    def __init__(self):
        super(LDC_Net_Cs, self).__init__() 
        self.FM = Feature_Extractor_R18() 
        self.classifier = Classifier()

    def forward(self, input):
        feature,catfeat = self.FM(input)
        classifier_out = self.classifier(feature)
        return classifier_out, feature
 

class ResNet_LS(nn.Module):
    def __init__(self, ):
        super(ResNet_LS, self).__init__() 
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

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal

class Estmator(nn.Module):
    def __init__(self, in_channels=448, out_channels=1, conv3x3=conv3x3):  # conv3x3 384 960 conv3x3_learn
        super(Estmator, self).__init__()

        self.conv = nn.Sequential(
            # nn.BatchNorm2d(in_channels), 
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

class LDC_Net_Cs_Dual(nn.Module):
    def __init__(self):
        super(LDC_Net_Cs_Dual, self).__init__() 
        self.FM = Feature_Extractor_R18() 
        self.classifier = Classifier()
        self.LiveEstor = Estmator()
        self.SpoofEstor = Estmator()

    def forward(self, input):
        feature,catfeat = self.FM(input)
        
        classifier_out = self.classifier(feature)
        live_Pre = self.LiveEstor(catfeat)
        spoof_Pre = self.SpoofEstor(catfeat)
        return classifier_out, feature,live_Pre,spoof_Pre

if __name__ == "__main__":
    net = LDC_Net_Cs_Dual().cuda()
    x = torch.randn(5, 3, 256, 256).cuda()
    net(x)
    # print(net)