from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import defaultdict

from torchvision.datasets import ImageFolder


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_no_corrects = 0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Set the model to the training mode for updating the weights using
        # the first portion of training images
        model.train()
        no_corrects = 0
        no_loss = 0.0
        for inputs, labels in dataloaders['train']:  # iterate over data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            no_loss += loss.item() * inputs.size(0)
            no_corrects += torch.sum(preds == labels.data)

        epoch_loss = no_loss / len(dataloaders['train'].dataset)
        epoch_acc = no_corrects.double() / len(dataloaders['train'].dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Set the model to the evaluation mode for selecting the best network
        # based on the number of correctly classified validation images
        model.eval()
        no_corrects = 0
        no_loss = 0.0
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                no_loss += loss.item() * inputs.size(0)
                no_corrects += torch.sum(preds == labels.data)
        if no_corrects > best_no_corrects:
            best_no_corrects = no_corrects
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

        epoch_loss = no_loss / len(dataloaders['valid'].dataset)
        epoch_acc = no_corrects.double() / len(dataloaders['valid'].dataset)
        print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_no_corrects / len(dataloaders['valid'].dataset)))

    # Load the weights of the best network
    model.load_state_dict(best_model_wts)
    return model


def make_weights_for_balanced_classes(images, nclasses):
    count = defaultdict(int)

    for _, label in images:
        count[label] += 1

    weight_per_class = [len(images) / count[i] for i in range(nclasses)]

    weights = [weight_per_class[label] for _, label in images]

    return weights


data_dir = 'dataset'

image_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

# Calculate weights only for the training set
train_weights = make_weights_for_balanced_classes(image_datasets['train'], 3)

dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=4,
        sampler=WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    ),
    'valid': DataLoader(image_datasets['valid'], batch_size=4, shuffle=True),
    'test': DataLoader(image_datasets['test'], batch_size=4, shuffle=True)
}


if __name__ == '__main__':

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

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)

    # Save the model

    # Create accuracy matrix
    with torch.no_grad():
        for type in ['train', 'valid', 'test']:
            confusion_matrix = torch.zeros(3, 3)
            for i, (inputs, classes) in enumerate(dataloaders[type]):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_conv(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            print(confusion_matrix)
            print(
                confusion_matrix.diag() / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - confusion_matrix.diag()))
            print(confusion_matrix.diag().sum() / confusion_matrix.sum())