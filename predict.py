import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
from utility import process_image, load_checkpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description= 'Parsing input to get image path, cat. mapping, top K classes, gpu')
    parser.add_argument('imagepath', default='flowers/test/15/image_06351.jpg')
    #The default checkpoint I used below is checkpoint.pth from first part of the project, based on instructions in
    #Part 2, to test with the checkpoint from part 1. I named the checkpoint generated from part 2 checkpnt.pth, 
    #and the checkpoint from part 1 checkpoint.pth . I tested both checkpoints, the one generated in part 1, and the
    #one generated in part 2, and both function correctly, give correct / proper predictions.
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth') 
    parser.add_argument('--top_k', dest='top_k', default='5', type=int)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    
    return parser.parse_args()

args = parse_arguments()


def loading_labels_to_names(filename):
    
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    
        return cat_to_name

def predict(image_path, model, topk=5, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
  
    if gpu == 'gpu'and torch.cuda.is_available(): 
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    
 
    with torch.no_grad():
        
        if gpu == 'gpu'and torch.cuda.is_available(): 
            image = image.to('cuda')
        else:
            image = image.to('cpu')
                          
        #from mentor's advice
        model.eval()
        
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        probabilities, indices = ps.topk(topk)
        if gpu == 'gpu' and torch.cuda.is_available():
            probabilities, indices = probabilities.to('cuda'), indices.to('cuda')
        else:
            probabilities, indices = probabilities.to('cpu'), indices.to('cpu')

        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        classes=[]
        indices = indices.cpu().numpy()        
        probabilities = probabilities.cpu().numpy()
        for index in indices[0]:
            classes.append(idx_to_class[index])
    
         
    return probabilities[0], classes

def main():
    args = parse_arguments()
    category_to_name = loading_labels_to_names(args.category_names)
    model = load_checkpoint(args.checkpoint)
    
    probabilities, classes = predict(args.imagepath, model, args.top_k, args.gpu)
    flower_names =[]
    for clss in classes:
        flower_names.append(category_to_name[str(clss)])    
           
    n = 0
    for n in range(len(flower_names)):
        print("Flower name: {}     Class Probability: {}".format(flower_names[n], probabilities[n]))
        n += 1

if __name__== "__main__":
    main()
        