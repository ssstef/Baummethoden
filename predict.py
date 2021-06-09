import pickle as pi
from sklearn.metrics import classification_report, confusion_matrix
from train import x_test,y_test,classifier,attribute_names
from sklearn.tree import export_graphviz
from subprocess import check_call

# Get file from pickle
infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()

# Prediction
y_pred_tree = classifier_dt.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_tree)
print(cm)

# ROC Curve
p1 = metrics.plot_roc_curve(classifier_dt, X_test, y_test)
plt.show()

##Rrandom Forest
# Random Forest implementieren

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)

# Prediction
y_pred_rf = classifier_rf.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)


# ROC Curve
metrics.plot_roc_curve(classifier_rf, X_test, y_test)
plt.show()

#Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# Prediction
y_pred_knn = classifier_knn.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)


# ROC Curve
metrics.plot_roc_curve(classifier_knn, X_test, y_test)
plt.show()

##SVM

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
classifier_dt
clf_svm = svm.SVC(gamma = 0.001, C=100, random_state=0)
clf_svm.fit(X_train, y_train)
#SVC(random_state=0)

#Roc Graph
metrics.plot_roc_curve(clf_svm, X_test, y_test)
plt.show()


#Kernel SVM

from sklearn.svm import SVC
classifier_ksvm = SVC(kernel = 'rbf', random_state = 0)
classifier_ksvm.fit(X_train, y_train)

# Prediction of Y
y_pred_ksvm = classifier.predict(X_test)

# Confusion Matrix
cm_ksvm = confusion_matrix(y_test, y_pred_ksvm)
print(cm)

#Roc Graph
metrics.plot_roc_curve(classifier_ksvm, X_test, y_test)
plt.show()

# Performance mittels ROC Graph vergleichen
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Show all plots in one figure

p1 = metrics.plot_roc_curve(classifier_dt, X_test, y_test)
p2 = metrics.plot_roc_curve(classifier_rf, X_test, y_test, ax=p1.ax_)
p3 = metrics.plot_roc_curve(classifier_knn, X_test, y_test, ax=p1.ax_)
p4 = metrics.plot_roc_curve(clf_svm, X_test, y_test, ax=p1.ax_)
plt.show()


##########
import pandas as pd

#Model aus Datei lesen
clf = pd.read_pickle(r'classifier_object.pickle')

data = pd.read_excel('data/Predict_Daten.xls')

x = data.loc[:, data.columns != 'class']

y_pred = clf.predict(x)

print(y_pred)


###############
import os
import pandas as pd
import pickle as pi

#save in Pickle file
path_start = os.getcwd()
pathr = os.path.dirname(os.getcwd())+'/../models'
os.chdir(pathr)
file_name = "classification_model.pickle"
fill = open(file_name,'rb')     #read only file in a Binary format
classifier = pi.load(fill)
fill.close()

#change to the start working directory
os.chdir(path_start)

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image']

pathread = os.path.dirname(os.getcwd())+'/../data/external/Predict.csv'
#Read csv0 -file
data = pd.read_csv(pathread, names=attribute_names)

#Get predicted values from test data 
y_pred = classifier.predict(data) 
#print(y_pred)

erg = pd.DataFrame(y_pred, columns = ['prediction'])
ergebnis = pd.concat([data,erg],axis=1)
print(ergebnis)

pathsave = os.path.dirname(os.getcwd())+'/../reports/prediction.xlsx'
ergebnis.to_excel(pathsave, sheet_name = 'Sheet_classification_divorce_1')