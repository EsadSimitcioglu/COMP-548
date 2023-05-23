from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_dir = 'dataset'

# Define the transformations for each dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Replace with actual mean values
                             std=[0.229, 0.224, 0.225]  # Replace with actual std values
                             )
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Replace with actual mean values
                             std=[0.229, 0.224, 0.225]  # Replace with actual std values
                             )
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Replace with actual mean values
                             std=[0.229, 0.224, 0.225]  # Replace with actual std values
                             )
    ])
}

# Load the datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'valid', 'test']}

def train_model(image_datasets, model, criterion, optimizer,
                scheduler, num_epochs):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_no_corrects = 0

    for epoch in range(num_epochs):
        # Set the model to the training mode for updating the weights using
        # the first portion of training images
        model.train()
        for inputs, labels in dataloaders['train']:  # iterate over data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Set the model to the evaluation mode for selecting the best network
        # based on the number of correctly classified validation images
        model.eval()
        no_corrects = 0
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                no_corrects += torch.sum(preds == labels.data)
        if no_corrects > best_no_corrects:
            best_no_corrects = no_corrects
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

    # Load the weights of the best network
    model.load_state_dict(best_model_wts)
    return model


# use the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Download the pretrained AlexNet model
model_conv = models.alexnet(pretrained=True)

# Freeze the parameters
for param in model_conv.parameters():
    param.requires_grad = False

# print(model_conv)

# Last layer of the AlexNet is Linear(in_features=4096, out_features=1000, bias=True)
# We need to change the last layer to have 3 classes instead of 1000
num_ftrs = model_conv.classifier[6].in_features
model_conv.classifier[6] = nn.Linear(num_ftrs, 3)
model_conv = model_conv.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(image_datasets, model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
