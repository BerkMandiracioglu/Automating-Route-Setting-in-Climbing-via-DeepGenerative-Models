###########################################################################
###########################################################################
"""CLASSES, FUNCTIONS AND LIBRARIES FOR REGRESSION AND CLASSIFICATION VIA SCIKIT AND NUMPY"""
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
from sklearn.model_selection import cross_val_score
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier

###########################################################################
###########################################################################
###########################################################################

"""CLASS FOR REGRESSION MODELS"""
class RegressorSklearn:
  
    """INITIALIZE: DO GRID SEARCH FOR THE REQUESTED REGRESSOR"""
    def __init__(self, selection = "linear"):
        self.selection = selection
        self.init = True
        if selection == "linear":
            """LINEAR REGRESSION"""
            parameters = {'fit_intercept':[True,False], 
                                'normalize':[True,False], 
                                'copy_X':[True, False]}
            self.model = GridSearchCV(LinearRegression() , param_grid = parameters, n_jobs=-1, cv=3)
        elif selection == "adaboost":
            """ADABOOST REGRESSION"""
            param_dist = {
                'n_estimators': [50, 100],
                'learning_rate' : [0.01,0.05,0.1],
                'loss' : ['linear', 'square', 'exponential']
                }
            self.model = GridSearchCV(AdaBoostRegressor(), param_grid = param_dist, n_jobs=-1, cv=3)
        
        elif selection == "randomforest":
            """RANDOM FOREST REGRESSION"""
            random_grid = {'n_estimators': [1,5,10,50],
                            'max_features': ['auto', 'sqrt'],
                            'max_depth': [10,20,50],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'bootstrap':  [True, False]
            }
            rf = RandomForestRegressor()

            self.model = GridSearchCV(estimator = rf, param_grid = random_grid, n_jobs=-1, cv=3)
        elif selection == "svm":
            """SUPPORT VECTOR MACHINE REGRESSION"""
            parameters_space = {'kernel': ('linear', 'rbf','poly'), 
                                'C':[1.5, 10],
                                'gamma': [1e-7, 1e-4],
                                'epsilon':[0.1,0.2,0.5,0.3]}
            self.model = GridSearchCV(svm.SVR(), param_grid = parameters_space, n_jobs=-1, cv=3)
        elif selection == "mlp":
            """MULTILAYER PERCEPTRON REGRESSION"""
            param_list = {
                    'hidden_layer_sizes': [(50),(100)],
                    'activation': ['relu'],
                    'solver': ['sgd','adam'],
                    'learning_rate_init': [0.001,0.005],
                    'learning_rate': ['adaptive'],
                }
            self.model = GridSearchCV(estimator=MLPRegressor(max_iter=5000), param_grid=param_list, n_jobs=-1, cv=3)
            #self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    """IMPORT DATA TO CLASS"""
    def import_data(self, x, y):
        
        self.x = x
        self.y = y
        self.x = self.x.reshape(self.x.shape[0],-1)
        self.y = self.y.reshape(self.y.shape[0])
        self.imported_ = True

    """SPLIT DATA INTO TRAIN AND VALIDATION"""
    def split(self, ratio = 0.20):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=ratio, shuffle=True)
        self.split_ = True
        


    """TRAIN REGRESSOR MODEL"""
    def train_regressor(self):
        self.start = time.time()    
        self.model.fit(self.x_train,self.y_train)
        self.end = time.time()
        self.best_params = self.model.best_params_
        
    """PREDICT LABEL"""
    def get_prediction(self, x_input):  
        self.predict = self.model.predict(x_input)
        self.predict = np.rint(self.predict)
        return self.predict

    """R2 SCORE FOR THE REGRESSOR"""
    def get_score(self):
        return (accuracy_score(self.y_test, self.get_prediction(self.x_test)), self.end - self.start, self.best_params)

###########################################################################
###########################################################################
###########################################################################

"""CLASS FOR CLASSIFICATION MODELS"""
class ClassifierSklearn:
 
    """INITIALIZE: DO GRID SEARCH FOR THE REQUESTED CLASSIFIER"""
    def __init__(self, selection = "svc"):
        self.selection = selection
        self.init = True

        if selection == "svc":
            """SUPPORT VECTOR MACHINE CLASSIFIER"""
            parameter_space={
                'C': [1, 10], 
                'kernel': ('linear', 'rbf')
            }
            svc = SVC(kernel = 'linear', C = 1)
            self.model = GridSearchCV(svc, parameter_space, n_jobs=-1, cv=3)

        elif selection == "decisiontree":
            """DECISION TREE CLASSIFIER"""
            dcs = DecisionTreeClassifier()
            parameter_space = {
                'criterion':['gini','entropy'],
                'max_depth' : [5, 10, 15]
            }
            self.model = GridSearchCV(dcs, parameter_space, n_jobs=-1, cv=3)

        elif selection == "knn":
            """K NEAREST NEIGHBORS CLASSIFIER"""
            knn = KNeighborsClassifier()
            parameter_space = {
                    'n_neighbors' : [5, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'metric':['euclidean','manhattan']
            }
            self.model = GridSearchCV(knn, parameter_space, n_jobs=-1, cv=3)

        elif selection == "mvc":
            """MULTILAYER PERCEPTRON CLASSIFIER"""
            mlp = MLPClassifier(max_iter=100)
            parameter_space = {
                    'hidden_layer_sizes': [(50),(100)],
                    'activation': ['relu'],
                    'solver': ['sgd','adam'],
                    'learning_rate_init': [0.001,0.005],
                    'learning_rate': ['adaptive'],
                }
            self.model = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)


    """IMPORT DATA TO CLASS"""
    def import_data(self, x, y):
        
        self.x = x
        self.y = y
        self.x = self.x.reshape(self.x.shape[0],-1)
        self.y = self.y.reshape(self.y.shape[0])
        self.imported_ = True

    """PREDICT LABEL"""
    def get_prediction(self, x_input):  
        return self.model.predict(x_input)

    """SPLIT DATA INTO TRAIN AND VALIDATION"""
    def split(self, ratio = 0.20):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=ratio, shuffle=True)
        self.split_ = True
        


    
    """TRAIN CLASSIFIER MODEL"""
    def train_classifier(self):
        self.start = time.time()
        self.model.fit(self.x_train,self.y_train)
        self.end = time.time()
        self.best_params = self.model.best_params_

    """VALIDATION SCORE"""
    def get_score(self):
        return (accuracy_score(self.y_test, self.get_prediction(self.x_test),normalize = False), self.end - self.start, self.best_params )

    
    """CROSSVALIDATION SCORE"""
    def get_cross_accuracy(self,k):
        return (np.mean(cross_val_score(self.model, self.x, self.y, cv=k)), self.end - self.start, self.best_params )


###########################################################################
###########################################################################
###########################################################################