from .convnet import *
from .resnet import *


def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

def Conv4S():
    return ConvNetS(4)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10(flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18(flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34(flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50(flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101(flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)
