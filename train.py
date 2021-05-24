# Pakete laden
import os
import sys  
from sklearn.tree import DecisionTreeClassifier
#from subprocess import check_call, check_output  # in Ruhe anschauen

# Set directory
current_dir = 'baummethoden'
#sys.path.append('../../src/features')  # noch anpassen

import pandas as pd
import numpy as np
import pickle as pi

import seaborn as sns
import matplotlib.pyplot as plt


import build_features
from build_features import *
direct = check_output('pwd') # ?
print(direct)
sys.path.append('../../src/visualization/')


# import visualize
# from visualize import *  ##selbst geschrieben?


import sklearn
from sklearn import preprocessing


# List with attribute names
attribute_names = []  # raussuchen!

#Read data
df = pd.read_excel('divorce.xlsx')
#data = pd.read_csv('../../../+current_dir+'/data/processed/dataset.csv, names = attribute_names)
# elegantere Methode verwenden.



# X und Y Variablen definieren
#'class'-column
y_variable = df['Class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
X_variables = df.loc[:, df.columns != 'Class']

X = X_variables
y = y_variable

# Train Test Split machen um Performance zu testen...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Normalisieren
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
print(sc.fit(X_train))
MinMaxScaler()
print(sc.data_max_)
print(sc.transform(X))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision Tree implementieren

classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

# Importance of attributes
classifier_dt.feature_importances_

# graphical output
Baum_tree(classifier_dt, attribute_names)

# save in Pickle file
path_start = os.getcwd()
pathr = os.path.dirnames(os.getcwd())+'/../models'
os.chdir(pathr)
file_name = "classification_divorce_pickle"
fill = open(file_name, 'wb')
pi.dump(classifier_dt, fill)
fill.close()

#Change to the start working directory
os.chdir(path_start)

