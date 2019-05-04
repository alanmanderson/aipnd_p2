import numpy as np

from PIL import Image

import torch
from torch.autograd import Variable

def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)

    tensor = torch.from_numpy(image).type(torch.FloatTensor)
    images = Variable(tensor, requires_grad=False).unsqueeze(0)
    images.to(device)

    ps = torch.exp(model.forward(images))

    top_p, top_labels = ps.topk(topk)
    top_p = top_p.data.numpy().squeeze()
    top_labels = top_labels.data.numpy().squeeze()

    idx_to_class = {}
    for cls, idx in model.class_to_idx.items():
        idx_to_class[idx] = cls

    classes = []
    for label in top_labels:
        classes.append(idx_to_class[label])

    return top_p, classes

DIMENSION = 224

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    if image.size[0] > image.size[1]:
        image.thumbnail((1000000, 256))
    else:
        image.thumbnail((256, 1000000))
    left = (image.width - DIMENSION) / 2
    top = (image.height - DIMENSION) / 2
    image = image.crop((left, top, left + DIMENSION, top + DIMENSION))
    image = np.array(image) / 255
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return image.transpose((2, 0, 1))
