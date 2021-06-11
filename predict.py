import pickle as pi
from sklearn.metrics import classification_report, confusion_matrix
from train import x_test,y_test,classifier,attribute_names
from sklearn.tree import export_graphviz
from subprocess import check_call

#Model aus Datei lesen
classifier_dt = pd.read_pickle(r'classifier_object.pickle')
data = pd.read_excel('data_banknote_authentication.csv')

x = data.loc[:, data.columns != 'class']
# Get file from pickle
#infile = open(clf_file,'rb')
#new_dict = pickle.load(infile)
#infile.close()

# Prediction
y_pred_tree = classifier_dt.predict(X_test)


import pandas as pd

#Model aus Datei lesen
clf = pd.read_pickle(r'classifier_object.pickle')

data = pd.read_excel('data/Predict_Daten.xls')

x = data.loc[:, data.columns != 'class']

y_pred = clf.predict(x)

print(y_pred)




#Model aus Datei lesen
clf = pd.read_pickle(r'classifier_object.pickle')

data = pd.read_excel('data/Predict_Daten.xls')

x = data.loc[:, data.columns != 'class']

y_pred = clf.predict(x)

print(y_pred)