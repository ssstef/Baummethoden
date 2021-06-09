# Pakete laden
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#Read data
df = pd.read_excel('divorce.xlsx')

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
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

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
