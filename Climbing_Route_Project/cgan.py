###########################################################################
###########################################################################
"""CLASSES AND LIBRARIES FOR CGAN"""
###########################################################################
###########################################################################
"""LIBRARIES REQUIRED"""
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
"""CGAN MODEL ARCHITECTURE: GENERATOR AND DISCRIMINATOR MODELS"""
class CGANGenerator(nn.Module):
    latent_space = 100
    def __init__(self,label_boi):
        super(CGANGenerator, self).__init__()
        self.fc1_1 = nn.Linear(100, 64)
        self.fc1_1_bn = nn.BatchNorm1d(64)
        self.fc1_2 = nn.Linear(label_boi, 64)
        self.fc1_2_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 198)

    #Weight initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    #Forward Prop.
    def forward(self, Input, Label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(Input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(Label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = torch.tanh(self.fc4(x))

        return x

class CGANDiscriminator(nn.Module):
    def __init__(self,label_boi):
        super(CGANDiscriminator, self).__init__()
        self.fc1_1 = nn.Linear(198, 128)
        self.fc1_2 = nn.Linear(label_boi, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)


    #Weight initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    #Forward Prop.
    def forward(self, Input, Label):
        x = F.leaky_relu(self.fc1_1(Input), 0.2)
        y = F.leaky_relu(self.fc1_2(Label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = torch.sigmoid(self.fc3(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

###########################################################################
###########################################################################
###########################################################################

"""CLASS TO TRAIN AND DEFINE CGAN"""
class TorchCGAN:

    latent_space = 100
    padded_size = 18
    model_type = "CGan"

    train_hist = {}
    train_hist['disc_losses'] = []
    train_hist['gen_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    def __init__(self,label_boi):
        self.init = True
        self.label_boi = label_boi

        self.class_labels = [i for i in range(self.label_boi)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Welcome to Climbing Route Generator!")
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
    def split(self, padding, ratio = 0.01):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=ratio, shuffle=True)
        self.split_ = True
    
        #ZEROPAD INSTEAD OF RESIZE
        data_transform = transforms.Compose([#transforms.Resize([32,32]),
                                            transforms.ToTensor()
                                            ])
        self.train = ClimbingDataset(self.x_train, self.y_train,padding,data_transform)
        self.test = ClimbingDataset(self.x_test, self.y_test,padding,data_transform)

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
        self.trainloader = torch.utils.data.DataLoader(self.train, batch_size=128)
        self.testloader = torch.utils.data.DataLoader(self.test, batch_size=128)

    '''CHOOSE MODEL: If model pretrained: pt_ = True'''
    '''MORE MODELS ARE TO BE INCLUDED'''
    def model_selection(self, model_name = "CGan", pt_=False):

        print("Model chosen as",model_name,"!")
        if model_name == "CGan":
            self.model_type = "CGan"
            self.generator_model = CGANGenerator(self.label_boi)
            self.discriminator_model = CGANDiscriminator(self.label_boi)
            self.generator_model.weight_init(mean=0.0, std=0.02)
            self.discriminator_model.weight_init(mean=0.0, std=0.02)
    

        if self.gpu_used:
            self.generator_model.cuda()
            self.discriminator_model.cuda()
            self.FloatTensor = torch.cuda.FloatTensor if self.gpu_used else torch.FloatTensor
            self.LongTensor = torch.cuda.LongTensor if self.gpu_used else torch.LongTensor
            self.ByteTensor = torch.cuda.ByteTensor if self.gpu_used else torch.ByteTensor


    '''OPTIMIZER AND CRITERION CHOICE'''
    def compile_model(self, lr_ = 0.0001):
        self.generator_model.to(self.device)
        self.discriminator_model.to(self.device)
        self.optimizer_gen = optim.Adam(self.generator_model.parameters(), lr = lr_, betas=(0.5, 0.999))
        self.optimizer_disc = optim.Adam(self.discriminator_model.parameters(), lr = lr_, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    '''TRAIN THE MODEL'''
    def train_model(self, path, epochs=20, batch_size=128):
        steps = 0
        best_model_gen = self.generator_model.state_dict()
        best_model_disc = self.discriminator_model.state_dict()

        epoch_loss = 2000.0

        for e in range(epochs):

            self.disc_losses = []
            self.gen_losses = []

            print('Epoch {}/{}'.format(e + 1, epochs))
            print('-' * 10)
            strt = time.time()
            self.discriminator_model.train(True)
            self.generator_model.train(True)

            if (e==5) or (e==10) or (e==25):
                self.optimizer_gen.param_groups[0]['lr'] /= 10
                self.optimizer_disc.param_groups[0]['lr'] /= 10
                print("Learning rate decreased!")

            valid = torch.ones(batch_size)
            fake = torch.zeros(batch_size)
            valid = valid - abs(torch.randn(batch_size)/10)
            fake = fake + abs(torch.randn(batch_size)/10)
        
        
            valid, fake = valid.to(self.device), fake.to(self.device)
      
            #self.scheduler.step()
            for inputs, labels in self.trainloader:
                '''TRAINING DISCRIMINATOR'''
                steps += 1
                # Move input and label tensors to the default device
            
                curr_size = inputs.size()[0] 
        
                if curr_size != batch_size:
                    valid = torch.ones(curr_size)
                    fake = torch.zeros(curr_size)

                    batch_size = curr_size
                                
                    valid = valid - abs(torch.randn(batch_size)/10)
                    fake = fake + abs(torch.randn(batch_size)/10)

                    valid, fake = valid.to(self.device), fake.to(self.device)

                '''TRAINING DISCRIMINATOR'''
                if steps % 41 == 0:
                    self.optimizer_disc.zero_grad()

                    #Check shape of label_Y
                    label_y = torch.zeros(batch_size, self.label_boi)
                    label_y.scatter_(1, labels.view(batch_size, 1), 1)

                    inputs = inputs.view(-1, 11 * 18)
                    inputs, label_y = inputs.to(self.device), label_y.to(self.device)
                    res_disc = self.discriminator_model(inputs, label_y).squeeze()
                    disc_valid_loss = self.criterion(res_disc, valid)

                    g1 = torch.rand((batch_size, self.latent_space))
                    labels = (torch.rand(batch_size, 1) * self.label_boi).type(torch.LongTensor)
                    label_y = torch.zeros(batch_size, self.label_boi)
                    label_y.scatter_(1, labels.view(batch_size, 1), 1)

                    g1, label_y = g1.to(self.device), label_y.to(self.device)

                    res_gen = self.generator_model(g1, label_y)
                    res_disc = self.discriminator_model(res_gen, label_y).squeeze()

                    disc_fake_loss = self.criterion(res_disc, fake)
                    disc_fake_score = res_disc.data.mean()

                    disc_train_loss = disc_valid_loss + disc_fake_loss

                    disc_train_loss.backward()
                    self.optimizer_disc.step()

                    self.disc_losses.append(disc_train_loss.item())

                '''TRAIN GENERATOR'''
                self.optimizer_gen.zero_grad()

                g1 = torch.rand((batch_size, self.latent_space))
                labels = (torch.rand(batch_size, 1) * self.label_boi).type(torch.LongTensor)
                label_y = torch.zeros(batch_size, self.label_boi)
                label_y.scatter_(1, labels.view(batch_size, 1), 1)

                g1, label_y = g1.to(self.device), label_y.to(self.device)

                res_gen = self.generator_model(g1, label_y)
                self.moonboard = res_gen.view(-1,11,18)
                self.moonboard = self.moonboard.view(-1,1,11,18)
                res_disc = self.discriminator_model(res_gen, label_y).squeeze()

                gen_train_loss = self.criterion(res_disc, valid)
                gen_train_loss.backward()
                self.optimizer_gen.step()

                self.gen_losses.append(gen_train_loss.item())
                self.last_batch_labels = labels

            end = time.time()
            per_epoch = end - strt
            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, loss_comb: %.3f' % (e + 1,(epochs), per_epoch, torch.mean(torch.FloatTensor(self.disc_losses)),
                                                            torch.mean(torch.FloatTensor(self.gen_losses)), (torch.mean(torch.FloatTensor(self.gen_losses)) + torch.mean(torch.FloatTensor(self.disc_losses)))/2))
      
            self.train_hist['disc_losses'].append(torch.mean(torch.FloatTensor(self.disc_losses)))
            self.train_hist['gen_losses'].append(torch.mean(torch.FloatTensor(self.gen_losses)))
            self.train_hist['per_epoch_ptimes'].append(per_epoch)
      
        print("Training finished! Saving results!")
        torch.save(self.generator_model.state_dict(), str(path)+"CGANgenerator_param"+str(self.label_boi)+".pkl")
        torch.save(self.discriminator_model.state_dict(), str(path)+"CGANdiscriminator_param"+str(self.label_boi)+".pkl")

        with open(str(path) + 'CGAN_train_hist.pkl', 'wb') as f:
            pickle.dump(self.train_hist, f)

        show_train_hist(self.train_hist, save=True, path=str(path)+'CGAN_train_hist'+str(self.label_boi)+'.png')
