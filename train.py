import argparse
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
import json
import PIL
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='train', description='Trains a classifier to identify images of flowers')

parser.add_argument('data_dir', type=str, help='The path to the data directory')
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--learning_rate', type=int, default=0.01)
parser.add_argument('--hidden_units', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--gpu_or_cpu', type=str, default='gpu', choices=['gpu', 'cpu'])

args = parser.parse_args()


# set up the path to the data directory

input_path = args.data_dir
if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()

data_dir = input_path
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# Transform and load data

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(validation_data, batch_size=64)


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = [train_transforms, validation_transforms, test_transforms]

# TODO: Load the datasets with ImageFolder
image_datasets = [train_data, validation_data, test_data]

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = [trainloader, validloader, testloader]


# The model

model = models.vargs.arch(pretrained=True)

# Cpu of Gpu

if args.gpu_or_cpu == 'gpu':
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# freeze the gradient on the model

for param in model.parameters():
    param.require_grad=False

model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

model.to(device);

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
