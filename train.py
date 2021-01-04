import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import utility

def parsing_arguments():
    
    parser = argparse.ArgumentParser(description ='Taking in the info needed to train the network from command line')
    parser.add_argument('data_dir', action='store', default='flowers')
    parser.add_argument('--save_dir', dest='save_dir', action='store', default = 'checkpnt.pth')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', type=float)
    parser.add_argument('--hidden_units', dest='hidden_units', default='512', type=int)
    parser.add_argument('--epochs', dest='epochs', default='7', type=int)
    parser.add_argument('--gpu', action='store', default='gpu')

    return parser.parse_args()

args = parsing_arguments()

def training(model, epochs, optimizer, criterion, trainloader, validloader, gpu):
    
    if gpu == 'gpu'and torch.cuda.is_available(): 
        model.to('cuda')
    else:
        model.to('cpu')
   
    with active_session():
        
        epochs = int(args.epochs)
        counter = 0
        running_loss = 0
        print_interval = 5
        
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if gpu == 'gpu'and torch.cuda.is_available(): 
            #model.to('cuda:0')
        #else:
            #model.to('cpu')

        for epoch in range(epochs):
            for images, labels in trainloader:
                counter += 1
               #images, labels = images.to(device), labels.to(device)
                if gpu == 'gpu'and torch.cuda.is_available(): 
                    images, labels = images.to('cuda'), labels.to('cuda')
                else:
                    images, labels = images.to('cpu'), labels.to('cpu')
                optimizer.zero_grad()
        
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if counter % print_interval == 0:
                    model.eval()
                    valid_loss = 0
                    valid_accuracy = 0
            
                    with torch.no_grad():
                        for images, labels in validloader:
                            #images, labels = images.to(device), labels.to(device)
                            if gpu == 'gpu' and torch.cuda.is_available(): 
                                images, labels = images.to('cuda'), labels.to('cuda')
                            else:
                                images, labels = images.to('cpu'), labels.to('cpu')
                            log_ps = model.forward(images)
                            batch_loss = criterion(log_ps, labels)
                    
                            valid_loss += batch_loss.item()
                    
                            ps = torch.exp(log_ps)
                            top_p, top_index = ps.topk(1, dim=1)
                            equals = top_index == labels.view(*top_index.shape)                     
                            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}    "
                          f"Training loss: {running_loss/print_interval:.3f}    "
                          f"Validation loss: {valid_loss/len(validloader):.3f}    "
                          f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

def save_checkpoint(model, PATH, optimizer, classifier, args):
    
    input_features = model.classifier[0].in_features
    
    checkpoint = {'network_type': args.arch, 
                  'model': model,
                  'epoch': args.epochs,
                  'input': input_features,
                  'hidden units': args.hidden_units,
                  'output': 102,
                  'classifier': classifier,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'learning_rate': args.learning_rate,
                  'print_interval': 5,
                  'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, PATH) 
    
def main():

    args = parsing_arguments()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ColorJitter(),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(45),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(500),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(500),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    #from this course's content
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   #if gpu == 'gpu'and torch.cuda.is_available(): 
       #model.to('cuda')
    model = getattr(models, args.arch)(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False
    
        if args.arch == "densenet121":
            classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(1024, args.hidden_units)),
                                      ('relu1', nn.ReLU()),
                                      ('dpout1', nn.Dropout(0.6)),                                          
                                      ('fc2', nn.Linear(args.hidden_units,102)),
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
        elif args.arch == "vgg13":
             classifier = nn.Sequential(OrderedDict([
                                       ('fc1', nn.Linear(25088, args.hidden_units)),
                                       ('relu1', nn.ReLU()),
                                       ('dpout1', nn.Dropout(0.6)),                                          
                                       ('fc2', nn.Linear(args.hidden_units,102)),
                                       ('output', nn.LogSoftmax(dim=1))
                                       ]))
        else:
            print("You cannot use {}. Please use either vgg13 or densenet121".format(args.arch))
                               
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    #model.to(device)
   #if gpu == 'gpu'and torch.cuda.is_available(): 
       #model.to('cuda')
    epochs = args.epochs
    gpu = args.gpu

    training(model, epochs, optimizer, criterion, trainloader, validloader, gpu)

    PATH = args.save_dir
    model.class_to_idx = train_data.class_to_idx

    save_checkpoint(model, PATH, optimizer, classifier, args)
    
if __name__=="__main__":
   main() 