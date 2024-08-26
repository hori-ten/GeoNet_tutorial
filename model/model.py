import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np


class Resnet(nn.Module):
  def __init__(self, resnet_n, pretrained=True):
    super().__init__()
    model_resnet = getattr(models, resnet_n)(pretrained=pretrained)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features



class Classifier(nn.Module):
    def __init__(self, feature_len, cate_num):
        super().__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class Encoder(nn.Module):
    def __init__(self, resnet, bn_dim=256, total_classes=None):
        super(Encoder, self).__init__()
        self.model_fc = Resnet(resnet)
        feature_len = self.model_fc.output_num()
        self.bottleneck_0 = nn.Linear(feature_len, bn_dim)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.BatchNorm1d(bn_dim), nn.ReLU())
        self.total_classes = total_classes
        if total_classes:
            self.classifier_layer = Classifier(bn_dim, total_classes)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        if not self.total_classes:
            return (out_bottleneck, None)
        logits = self.classifier_layer(out_bottleneck)
        return (out_bottleneck, logits)
