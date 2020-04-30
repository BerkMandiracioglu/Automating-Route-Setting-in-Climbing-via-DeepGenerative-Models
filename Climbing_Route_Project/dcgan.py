###########################################################################
###########################################################################
"""CLASSES AND LIBRARIES REQUIRED FOR DCGAN"""
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
"""DCGAN MODEL ARCHITECTURE: GENERATOR AND DISCRIMINATOR"""
class DCGANGenerator(nn.Module):
    latent_space = 50
    def __init__(self, label_boi, d=128):
        super(DCGANGenerator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(self.latent_space, int(d*2), 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(int(d*2))
        self.deconv1_2 = nn.ConvTranspose2d(label_boi, int(d*2), 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(int(d*2))
        self.deconv2 = nn.ConvTranspose2d(int(d*4), int(d*2), 2, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(int(d*2))
        self.deconv3 = nn.ConvTranspose2d(int(d*2), int(d), 2, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(int(d))
        self.deconv4 = nn.ConvTranspose2d(int(d), 1, (4, 3), (1,2), 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, Input, Label):
        
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(Input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(Label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))

        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self, label_boi, d=128):
        super(DCGANDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(label_boi, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, int(d*2), 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(int(d*2))
        self.conv3 = nn.Conv2d(int(d*2), int(d*4), 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(int(d*4))
        self.conv4 = nn.Conv2d(int(d*4), 1, 1, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, Input, Label):
        x = F.leaky_relu(self.conv1_1(Input), 0.2)
        y = F.leaky_relu(self.conv1_2(Label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
    
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


###########################################################################
###########################################################################
###########################################################################

"""CLASS TO TRAIN AND DEFINE DCGAN"""
class TorchDCGAN:

    latent_space = 50
    padded_size = 18
    model_type = "DCGan"

    train_hist = {}
    train_hist['disc_losses'] = []
    train_hist['gen_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    def __init__(self, label_boi):
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
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train = ClimbingDataset(self.x_train, self.y_train, padding,data_transform)
        self.test = ClimbingDataset(self.x_test, self.y_test, padding,data_transform)

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
        self.trainloader = torch.utils.data.DataLoader(self.train, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(self.test, batch_size=64)

    '''CHOOSE MODEL: If model pretrained: pt_ = True'''
    '''MORE MODELS ARE TO BE INCLUDED'''
    def model_selection(self, model_name = "DCGan", pt_=False):

        print("Model chosen as",model_name,"!")
        if model_name == "DCGan":
            self.model_type = "DCGan"
            self.generator_model = DCGANGenerator(self.label_boi)
            self.discriminator_model = DCGANDiscriminator(self.label_boi)
            self.generator_model.weight_init(mean=0.0, std=0.02)
            self.discriminator_model.weight_init(mean=0.0, std=0.02)
    

        if self.gpu_used:
            self.generator_model.cuda()
            self.discriminator_model.cuda()
            self.FloatTensor = torch.cuda.FloatTensor if self.gpu_used else torch.FloatTensor
            self.LongTensor = torch.cuda.LongTensor if self.gpu_used else torch.LongTensor
            self.ByteTensor = torch.cuda.ByteTensor if self.gpu_used else torch.ByteTensor

    '''OPTIMIZER AND CRITERION CHOICE'''
    def compile_model(self, lr_ = 0.00001):
        self.generator_model.to(self.device)
        self.discriminator_model.to(self.device)
        self.optimizer_gen = optim.Adam(self.generator_model.parameters(), lr = lr_, betas=(0.5, 0.999))
        self.optimizer_disc = optim.Adam(self.discriminator_model.parameters(), lr = lr_, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()


    def train_model(self,path, epochs=20, batch_size=64):
        steps = 0
        best_model_gen = self.generator_model.state_dict()
        best_model_disc = self.discriminator_model.state_dict()   

        # label preprocess
        onehot = torch.zeros(self.label_boi, self.label_boi)  
        onehot = onehot.scatter_(1, torch.LongTensor(self.class_labels).view(self.label_boi,1), 1).view(self.label_boi, self.label_boi, 1, 1)
        fill = torch.zeros([self.label_boi, self.label_boi, 11, 18])
        for i in range(self.label_boi):
            fill[i,i, :, :] = 1
    
        for e in range(epochs):
        
            if (e+1)==5 or (e+1)==10 or (e+1) == 25:
                self.optimizer_gen.param_groups[0]['lr'] /= 10
                self.optimizer_disc.param_groups[0]['lr'] /= 10
                print("learning rate decreased")
            
            self.disc_losses = []
            self.gen_losses = []
            print('Epoch {}/{}'.format(e + 1, epochs))
            print('-' * 10)
            strt = time.time()
            self.discriminator_model.train(True)
            self.generator_model.train(True)
            valid = torch.ones(int(batch_size/2),2,2)
            fake = torch.zeros(int(batch_size/2),2,2)
            valid = valid - abs(torch.randn(int(batch_size/2),2,2)/10)
            fake = fake + abs(torch.randn(int(batch_size/2),2,2)/10)
            
            '''
            rand_tmp = np.random.randint(batch_size,size=batch_size//8)
            valid[rand_tmp] = 0
            fake[rand_tmp] = 1
            '''
            
            valid, fake = valid.to(self.device), fake.to(self.device)

            for inputs, labels in self.trainloader:
                '''TRAINING DISCRIMINATOR'''
                steps += 1
                # Move input and label tensors to the default device     
                curr_size = inputs.size()[0] 

                if curr_size == batch_size:

                    if steps % 41 == 0:
                        self.optimizer_disc.zero_grad()

                        labels_fill = fill[labels]
                        inputs, labels_fill = inputs.to(self.device), labels_fill.to(self.device)
                        #inputs = inputs.squeeze()
                        
                        res_disc = self.discriminator_model(inputs, labels_fill).squeeze()
                        self.disc_loss_valid = self.criterion(res_disc, valid)

                        g1= torch.randn((batch_size, self.latent_space)).view(-1, self.latent_space, 1, 1)
                        y_tmp = (torch.rand(batch_size, 1) * self.label_boi).type(torch.LongTensor).squeeze()
                        label_y = onehot[y_tmp]
                        labels_fill = fill[y_tmp]
                        g1, label_y, labels_fill = g1.to(self.device), label_y.to(self.device), labels_fill.to(self.device)

                        res_gen = self.generator_model(g1, label_y)
                        self.moonboard = res_gen
                        res_disc = self.discriminator_model(res_gen, labels_fill).squeeze()

                        self.disc_loss_fake = self.criterion(res_disc, fake)
                        self.disc_score_fake = res_disc.data.mean()

                        self.disc_train_loss = self.disc_loss_valid + self.disc_loss_fake

                        self.disc_train_loss.backward()
                        self.optimizer_disc.step()

                        #print(self.disc_train_loss.item())
                        self.disc_losses.append(self.disc_train_loss.item())
                    
                    '''TRAINING GENERATOR'''
                    
                    self.optimizer_gen.zero_grad()

                    g1= torch.randn((batch_size, self.latent_space)).view(-1, self.latent_space, 1, 1)
                    y_tmp = (torch.rand(batch_size, 1) * self.label_boi).type(torch.LongTensor).squeeze()
                    label_y = onehot[y_tmp]
                    labels_fill = fill[y_tmp]
                    g1, label_y, labels_fill = g1.to(self.device), label_y.to(self.device), labels_fill.to(self.device)

                    res_gen = self.generator_model(g1, label_y)
                    res_disc = self.discriminator_model(res_gen, labels_fill).squeeze()

                    self.gen_train_loss = self.criterion(res_disc, valid)
                    self.gen_train_loss.backward()
                    self.optimizer_gen.step()

                    self.gen_losses.append(self.gen_train_loss.item())
                    self.last_batch_labels = labels

            end = time.time()
            per_epoch = end - strt
            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, loss_comb: %.3f' % (e + 1,(epochs), per_epoch, torch.mean(torch.FloatTensor(self.disc_losses)),
                                                            torch.mean(torch.FloatTensor(self.gen_losses)), (torch.mean(torch.FloatTensor(self.gen_losses)) + torch.mean(torch.FloatTensor(self.disc_losses)))/2))
            self.train_hist['disc_losses'].append(torch.mean(torch.FloatTensor(self.disc_losses)))
            self.train_hist['gen_losses'].append(torch.mean(torch.FloatTensor(self.gen_losses)))
            self.train_hist['per_epoch_ptimes'].append(per_epoch)
        
        print("Training finished! Saving results!")
        torch.save(self.generator_model.state_dict(), str(path)+"DCGANgenerator_param"+str(self.label_boi)+".pkl")
        torch.save(self.discriminator_model.state_dict(), str(path)+"DCGANdiscriminator_param"+str(self.label_boi)+".pkl")

        with open(str(path) + 'DC_train_hist.pkl', 'wb') as f:
            pickle.dump(self.train_hist, f)

        show_train_hist(self.train_hist, save=True, path=str(path)+'DCGAN_train_hist'+str(self.label_boi)+'.png')
