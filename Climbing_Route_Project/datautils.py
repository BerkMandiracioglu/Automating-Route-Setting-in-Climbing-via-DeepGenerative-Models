###########################################################################
###########################################################################
"""CLASSES, FUNCTIONS AND LIBRARIES FOR DATA MANIPULATION"""
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


###########################################################################
###########################################################################
###########################################################################
class ClimbingDataset(Dataset):
    'DATASET FOR PYTORCH'
    def __init__(self, x, y, padding, transform = None):
        'INITILIAZATION'
        if padding:
            self.shape = (x.shape[0],18,18)
            self.x = np.zeros((self.shape[0],self.shape[1], self.shape[2]))
            self.x[:,:x.shape[1],:x.shape[2]] = x
        else:
            self.x = x

        self.y = y
        self.transform = transform

    def __len__(self):
        '# OF SAMPLES'
        return len(self.x)

    def __getitem__(self, index):
        'SAMPLE OF DATA'
        # Select sample
        X = self.x[index]
        Y = self.y[index]
    
        #Convert to grayscale
        X = Image.fromarray(X, 'L')
        
        if self.transform is not None:
            X = self.transform(X)

        return X, Y

###########################################################################
###########################################################################
###########################################################################

"""CREATE DATA BOARDS"""
def moveToArray(lst):
    """Function parsing "moves" data structure into 11x18x3 tensor"""
    board = np.zeros((3,11,18))
    data = np.zeros((11,18))
    for string in lst:
        splt = string.split(" ")
        x = ord(splt[0][0])-65
        y = int(splt[0][1:]) - 1
        if splt[1][0] == 'T':
            board[0][x][y] = 1
            data[x][y] = 1
        elif splt[2][0] == 'T':
            board[2][x][y] = 1          
            data[x][y] = 3
        else:
            board[1][x][y] = 1
            data[x][y] = 2
    return (board, data)

###########################################################################
###########################################################################
###########################################################################

'''MODE APPLIED: MODE 0 = 3 CLASS; MODE 1 = 8 CLASS MODE 2 = 16 CLASS'''
def grade2Label(string,mode=0):
    """Function converting climbing grades into discrete labels"""
    if string == "6A":
        return 0

    elif string == "6A+":
        if mode == 0 or mode == 1:
            return 0
        elif mode == 2:
            return 1

    elif string == "6B":
        if mode == 0:
            return 0
        elif mode == 1:
            return 1
        elif mode == 2:
            return 2

    elif string == "6B+":
        if mode == 0:
            return 0
        elif mode == 1:
            return 1
        elif mode == 2:
            return 3

    elif string == "6C":
        if mode == 0:
            return 0
        elif mode == 1:
            return 2
        elif mode == 2:
            return 4

    elif string == "6C+":
        if mode == 0:
            return 0
        elif mode == 1:
            return 2
        elif mode == 2:
            return 5

    elif string == "7A":
        if mode == 0:
            return 1
        elif mode == 1:
            return 3
        elif mode == 2:
            return 6

    elif string == "7A+":
        if mode == 0:
            return 1
        elif mode == 1:
            return 3
        elif mode == 2:
            return 7

    elif string == "7B":
        if mode == 0:
            return 1
        elif mode == 1:
            return 4
        elif mode == 2:
            return 8

    elif string == "7B+":
        if mode == 0:
            return 1
        elif mode == 1:
            return 4
        elif mode == 2:
            return 9

    elif string == "7C":
        if mode == 0:
            return 1
        elif mode == 1:
            return 5
        elif mode == 2:
            return 10

    elif string == "7C+":
        if mode == 0:
            return 1
        elif mode == 1:
            return 5
        elif mode == 2:
            return 11

    elif string == "8A":
        if mode == 0:
            return 2
        elif mode == 1:
            return 6
        elif mode == 2:
            return 12

    elif string == "8A+":
        if mode == 0:
            return 2
        elif mode == 1:
            return 6
        elif mode == 2:
            return 13

    elif string == "8B":
        if mode == 0:
            return 2
        elif mode == 1:
            return 7
        elif mode == 2:
            return 14

    elif string == "8B+":
        if mode == 0:
            return 2
        elif mode == 1:
            return 7
        elif mode == 2:
            return 15

    else:
        return 0

###########################################################################
###########################################################################
###########################################################################       

'''MODE APPLIED: MODE 0 = 3 CLASS; MODE 1 = 7 CLASS MODE 2 = 14 CLASS'''    
def label2Grade(label, mode=0):
    """Function converting climbing grades into discrete labels"""
    if mode == 0:
        if label == 0:
            return "6[C - A+]"
        elif label == 1:
            return "7[C - A+]"
        elif label == 2:
            return "8[B - A+]"
        else:
            return "Label not valid!"
    if mode == 1:
        if label == 0:
            return "6A - 6A+"
        elif label == 1:
            return "6B - 6B+"
        elif label == 2:
            return "6C - 6C+"
        elif label == 3:
            return "7A - 7A+"
        elif label == 4:
            return "7B - 7B+"
        elif label == 5:
            return "7C - 7C+"
        elif label == 6:
            return "8A - 8A+"
        elif label == 7:
            return "8B - 8B+"
        else:
            return "Label not valid!"

    if mode == 2:
        if label == 0:
            return "6A"
        elif label == 1:
            return "6A+"
        elif label == 2:
            return "6B"
        elif label == 3:
            return "6B+"
        elif label == 4:
            return "6C"
        elif label == 5:
            return "6C+"
        elif label == 6:
            return "7A"
        elif label == 7:
            return "7A+"
        elif label == 8:
            return "7B"
        elif label == 9:
            return "7B+"
        elif label == 10:
            return "7C"
        elif label == 11:
            return "7C+"
        elif label == 12:
            return "8A"
        elif label == 13:
            return "8A+"
        elif label == 14:
            return "8B"
        elif label == 15:
            return "8B+"
        else:
            return "Label not valid!"

###########################################################################
###########################################################################
###########################################################################
      



###########################################################################
###########################################################################
###########################################################################
'''PLOT LOSS GRAPHS FOR GANS'''
def show_train_hist(hist, show = False, save = False, path = 'drive/My Drive/Generative_Climbing/Train_hist.png'):
    x = range(len(hist['disc_losses']))

    y1 = hist['disc_losses']
    y2 = hist['gen_losses']

    plt.plot(x, y1, label='Discriminator_loss')
    plt.plot(x, y2, label='Generator_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

###########################################################################
###########################################################################
###########################################################################
'''PLOT LOSS GRAPHS FOR PYTORCH CLASSIFIER'''
def show_train_hist_classifier(hist, show = False, save = False, path = 'drive/My Drive/Generative_Climbing/Train_hist.png'):
    x = range(len(hist['classifier_acc']))

    y1 = hist['classifier_acc']
    y2 = hist['validation_acc']

    plt.plot(x, y1, label='Classifier_acc')
    plt.plot(x, y2, label='Validation_acc')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

###########################################################################
###########################################################################
###########################################################################
'''CONVERT FLOAT RESULTS TO ACTUAL HOLDING POINTS'''
def convert_output(out, threshold,model):
    prob = out*0.5 + 0.5
    t = Variable(torch.Tensor([threshold]))
    t = t.to(model.device)

    out = (prob > t).float() 
  
    if out.shape[2] == 18:
        out = out[:,:,:].cpu().data.numpy()
    else:
        out = out[:,:,:18].cpu().data.numpy()

    nonzeros_row = np.nonzero(out[0])[0]
    nonzeros_col = np.nonzero(out[0])[1]
    min_val = min(nonzeros_col)
    max_val = max(nonzeros_col)
  
    dict_ind_first = {}
    dict_ind_last = {}
    for i in range(nonzeros_row.shape[0]):
        if(nonzeros_col[i] == min_val):
            dict_ind_first[nonzeros_row[i]] = min_val
        elif(nonzeros_col[i] == max_val):
            dict_ind_last[nonzeros_row[i]] = max_val

    zeros_first = np.zeros((1,11,18))
    zeros_last = np.zeros((1,11,18))

    for key in dict_ind_first:  
        zeros_first[0][key, dict_ind_first[key]] = 1
        out[0][key, dict_ind_first[key]] = 0

    for key in dict_ind_last:
        zeros_last[0][key, 17] = 1
        out[0][key, 17] = 0


  
    newout = np.vstack((zeros_first,out))
    newout = np.vstack((newout,zeros_last))
    return newout

###########################################################################
###########################################################################
###########################################################################
"""VISUALIZE SAMPLES USING OPENCV"""
def visualize_sample(x,y, mode = "2017", label_mode = 0):
    "Utils to dispaly moonboard problems"

    if mode == "2017":
        img = cv2.imread('/content/drive/My Drive/Climbing_Route_Project/background2017.jpg')
        H,W,_ = img.shape
        pH_top = 20
        pH_bottom = 35
        pW_left = 52
        pW_right = 24
    else:
        img = cv2.imread('/content/drive/My Drive/Climbing_Route_Project/background2016.jpg')
        H,W,_ = img.shape
        pH_top = 20
        pH_bottom = 35
        pW_left = 46
        pW_right = 0
    
    # define padding based on image
    H = H-pH_top -pH_bottom
    W = W-pW_left - pW_right
    
    # colors for start holds, mid holds and top holds
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    
    for i in range(x.shape[0]):
        holds_x, holds_y = np.nonzero(x[i,:,:])
        for j in range(len(holds_x)):
            # now draw holds
            cx = pW_left+int( float(holds_x[j])/x.shape[1]*W)
            cy = pH_top+int(H - float(holds_y[j])/x.shape[2]*H)
            cv2.circle(img=img, center=(cx,cy), radius=10, color=colors[i], thickness=2)
    
    plt.figure(figsize=(10,10))
    plt.imshow(img[:,:,::-1])
    plt.title(label2Grade(y, label_mode))
    
'''VISUALIZE RESULTS'''
def view_results(gen_output, threshold, itercount,model, labels, label_mode):
    iters = 0
    for image in gen_output:
        newout = convert_output(image, threshold,model)
        visualize_sample(newout, labels[iters] ,mode = "2017",label_mode = label_mode)
        iters += 1
        if iters == itercount:
            break

def GANQuantity(gan_labels, gan_predictions):
    """GAN LABEL VS PREDICTED LABEL ACCURACY CALCULATION"""
    total = gan_labels.shape[0]
    sum_gan = 0
    for i in range(gan_labels.shape[0]):
        if gan_labels[i] == gan_predictions[i]:
            sum_gan += 1
    res =  sum_gan / total
    return res

def LabelGANResults(torch_cgan, torch_dcgan, clf , sklearn_flag):
    """FOR CGAN RESULTS"""
    if sklearn_flag:
        test_images_c = torch_cgan.moonboard.cpu().detach()
        test_images_c = test_images_c.reshape(test_images_c.shape[0],-1)
        test_images_c = test_images_c.numpy()
    
    else:
        test_images_c = torch_cgan.moonboard
  
    prediction = clf.get_prediction(test_images_c)
 
    res = np.zeros(prediction.shape[0])
    labels = []
    labels2 = []
    for i in range(prediction.shape[0]):
        for j in range(torch_cgan.label_boi):  
            if sklearn_flag:
                res[i] = prediction[i]
                break
            if max(prediction[i]) == prediction[i,j]:
                res[i] = j
        labels.append(grade2Label(res[i]))
    """FOR DCGAN RESULTS"""
    if sklearn_flag:
        test_images_dc = torch_dcgan.moonboard[:,:,:,:18].cpu().detach()
        test_images_dc = test_images_dc.reshape(test_images_dc.shape[0],-1)
        test_images_dc = test_images_dc.numpy()
    else:
        test_images_dc = torch_dcgan.moonboard[:,:,:,:18]
  
    prediction2 = clf.get_prediction(test_images_dc)
    res2 = np.zeros(prediction2.shape[0])
    for i in range(prediction2.shape[0]):
        for j in range(torch_dcgan.label_boi):
            if sklearn_flag:
                res2[i] = prediction[i]
                break
            if max(prediction2[i]) == prediction2[i,j]:
                res[i] = j
        labels2.append(grade2Label(res2[i]))

    return labels, labels2, torch_cgan.moonboard, torch_dcgan.moonboard[:,:,:,:18]
