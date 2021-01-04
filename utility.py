import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import json


def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    changes_to_image = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
 
    image_tensor = changes_to_image(pil_image)
    return image_tensor

    
def load_checkpoint(PATH):
    
    checkpoint = torch.load(PATH)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer_state_dict']
    lr = checkpoint['learning_rate']
    print_interval = checkpoint['print_interval']
    epochs = checkpoint['epoch']
    
    for parameter in model.parameters():
        parameter.requires_grad=False
            
    return model