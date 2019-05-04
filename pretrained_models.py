from torchvision import models
from torch import nn
from collections import OrderedDict

def make_model(arch, layer_sizes, out_size):
    model, in_layers = get_pretrained_model(arch)
    features = list(model.classifier.children())[:-1]
    in_layers = model.classifier[len(features)].in_features

    features.extend([
        nn.Dropout(),
        nn.Linear(in_layers, layer_sizes),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(layer_sizes, layer_sizes),
        nn.ReLU(True),
        nn.Linear(layer_sizes, out_size),
    ])
    model.classifier = nn.Sequential(*features)
    #layers = []
    #layer_sizes.insert(0, in_layers)
    #for i in range(len(layer_sizes) - 2):
    #    layers.append(('fc' + str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1])))
    #    layers.append(('relu' + str(i), nn.ReLU()))
    #    layers.append(('dropout' + str(i), nn.Dropout(p=0.5)))
    #layers.append(('output', nn.Linear(layer_sizes[-2], layer_sizes[-1])))
    #layers.append(('softmax', nn.LogSoftmax(dim=1)))
    #model.classifier = nn.Sequential(OrderedDict(layers))
    return model

def get_pretrained_model(arch):
    if arch == 'vgg19':
        return models.vgg19(pretrained=True), 25088
    if arch == 'vgg16':
        return models.vgg16(pretrained=True), 25088
    if arch == 'vgg13':
        return models.vgg13(pretrained=True), 25088
    if arch == 'resnet101':
        return models.resnet18(pretrained=True), 2048
    if arch == 'alexnet':
        return models.alexnet(pretrained=True), 9216
    raise Exception("Unsupported architecture: " + arch)
