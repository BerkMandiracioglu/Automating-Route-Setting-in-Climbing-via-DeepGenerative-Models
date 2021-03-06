###########################################################################
###########################################################################
"""CLASSES, FUNCTIONS AND LIBRARIES FOR CLASSIFICATION"""
###########################################################################
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getpass
import os
import sys
import io
import time
import cv2
import pickle
import imageio
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.distributions as tdist
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from datautils import *

###########################################################################
###########################################################################
###########################################################################

"""PYTORCH CLASSIFIER MODEL ARCHITECTURE"""
class Classifier(nn.Module):
    def __init__(self, label_boi):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(198,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256, label_boi)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.softmax(x, dim=1)
        return output

###########################################################################
###########################################################################
###########################################################################

"""PYTORCH CLASSIFIER MODEL TRAINING AND DEFINITION"""
class TorchClassifier:
    train_hist = {}
    train_hist['classifier_acc'] = []
    train_hist['validation_acc'] = []
    accuracies = []
    valid_acc = []
    def __init__(self, label_boi):
        self.init = True
        self.label_boi = label_boi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Welcome to Climbing Route Classifier!")
        if torch.cuda.is_available():
            self.gpu_used = True
            print("GPU mode activated!(AFTER MODEL SELECTION)")
        else:
            print("CPU mode activated!(AFTER MODEL SELECTION)")

    '''IMPORT DATA TO CLASS'''
    def import_data(self, x, y):
        self.x = x
        self.y = y
        self.imported_ = True

    '''SPLIT DATA INTO TRAIN AND VALIDATION'''
    def split(self, padding, ratio = 0.1):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=ratio, shuffle=True)
        self.split_ = True
        
        data_transform = transforms.Compose([transforms.ToTensor()])
        self.train = ClimbingDataset(self.x_train, self.y_train, padding, data_transform)
        self.test = ClimbingDataset(self.x_test, self.y_test, padding, data_transform)
     
    '''DISPLAY FULL DATA'''
    def display_full_data(self):
        if not self.imported_:
            print("Data has not been provided yet!: Use .import_data(x,y)")
        else:
            print("X shape: ", self.x.shape)
            print("Y shape: ", self.y.shape)

    '''DISPLAY SPLIT DATA'''
    def display_split_data(self):
        if not self.split_:
            print("Data has not been split yet!: Use .split()")
        else:
            print("X_train shape: ", self.x_train.shape)
            print("Y_train shape: ", self.y_train.shape)
            print("X_test shape: ", self.x_test.shape)
            print("Y_test shape: ", self.y_test.shape)

    def data_to_loader(self):
        self.trainloader = torch.utils.data.DataLoader(self.train, batch_size=32)
        self.testloader = torch.utils.data.DataLoader(self.test, batch_size=32)

    '''CHOOSE MODEL: If model pretrained: pt_ = True'''
    '''MORE MODELS ARE TO BE INCLUDED'''
    def model_selection(self, model_name = "custom", input_shape = (11,18), pt_=False):

        print("Model chosen as",model_name,"!")
        if model_name == "resnet":
            self.model = models.resnet152(pretrained=pt_)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.label_boi)
        elif model_name == "vgg":
            self.model = models.vgg19(pretrained=pt_)
            self.model.fc = nn.Linear(2048, self.label_boi)
        elif model_name == "custom":
            self.model = Classifier(self.label_boi)

        if self.gpu_used:
            self.model.cuda()
            self.FloatTensor = torch.cuda.FloatTensor if self.gpu_used else torch.FloatTensor
            self.LongTensor = torch.cuda.LongTensor if self.gpu_used else torch.LongTensor
            self.ByteTensor = torch.cuda.ByteTensor if self.gpu_used else torch.ByteTensor


    '''OPTIMIZER AND CRITERION CHOICE'''
    def compile_model(self, lr_ = 0.0000001):
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_)
        self.criterion = nn.CrossEntropyLoss()
  
    '''PREDICT RESULTS'''
    def get_prediction(self,board):
        return self.model(board)

    '''TRAINING PHASE'''
    def train_model(self, path, epochs = 10):
        
        steps = 0
        since = time.time()
        self.best_model_wts = self.model.state_dict()
        self.best_test_acc = 0.0
        epoch_test_acc = 0.0

        for e in range(epochs):
            print('Epoch {}/{}'.format(e + 1, epochs))
            print('-' * 10)
            since_epoch = time.time()
            accuracies = []
            valid_acc = []
            self.model.train(True)

            running_loss = 0.0
            running_corrects = 0
            
            if ((e + 1) % 15  == 0):
                self.optimizer.param_groups[0]['lr'] /= 10
                print("Learning rate decreased!")

            for inputs, labels in self.trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs = Variable(inputs.type(self.FloatTensor))
                labels = Variable(labels.type(self.LongTensor))
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                
                self.loss = self.criterion(outputs, labels)
                self.loss.backward()
                self.optimizer.step()
            
                running_loss += self.loss.item()
                
                running_corrects += torch.sum(preds.data == labels.data).item()

            epoch_loss = running_loss / len(self.train)
            epoch_acc = running_corrects / len(self.train)
            self.accuracies.append(epoch_acc)
            time_elapsed_epoch = time.time() - since_epoch
            print('Loss: {:.4f} Acc: {:.4f} m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed_epoch // 60))
            
            
            self.model.eval()
            with torch.no_grad():
                testing_loss = 0.0
                test_corrects = 0
                for test_inputs, test_labels in self.testloader:
                    test_inputs = Variable(test_inputs.type(self.FloatTensor))
                    test_labels = Variable(test_labels.type(self.LongTensor))
                    test_inputs, test_labels = test_inputs.to(self.device), test_labels.to(self.device)
                                
                    test_outputs = self.model(test_inputs)
                    _, test_preds = torch.max(test_outputs.data, 1)
                    test_loss = self.criterion(test_outputs, test_labels)
                                
                    testing_loss += test_loss.item()
                    test_corrects += torch.sum(test_preds.data == test_labels.data).item()
                    
                epoch_test_loss = testing_loss / len(self.test)
                epoch_test_acc = test_corrects / len(self.test)
                print('Validation Loss: {:.4f} Validation Acc: {:.4f}'.format(epoch_test_loss, epoch_test_acc))   
                self.valid_acc.append(epoch_test_acc)

                if epoch_test_acc > self.best_test_acc:
                    print("Saved model state: Validation accuracy performed better!")
                    self.best_test_acc = epoch_test_acc
                    self.best_model_wts = self.model.state_dict()
            
            self.train_hist['classifier_acc'].append(torch.mean(torch.FloatTensor(self.accuracies)))
            self.train_hist['validation_acc'].append(torch.mean(torch.FloatTensor(self.valid_acc)))

        '''SAVE BEST MODEL WEIGHTS'''
        print("Training finished! Saving results!")
        torch.save(self.best_model_wts, str(path)+"classifier_param"+str(self.label_boi)+".pkl")

        with open(str(path) + 'train_hist.pkl', 'wb') as f:
            pickle.dump(self.train_hist, f)

        show_train_hist_classifier(self.train_hist, save=True, path=str(path)+'train_hist'+str(self.label_boi)+'.png')


###########################################################################
###########################################################################
###########################################################################
