from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import mltools as ml
from sklearn import svm
from sklearn.externals import joblib


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC



from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

x_data = np.genfromtxt("/data/users/anuragm/X_train.txt",delimiter=None)
y_data = np.genfromtxt("/data/users/anuragm/Y_train.txt",delimiter=None)

x_data = x_data[:, 0:13]
#Xtr, Ytr = ml.shuffleData(x_data, y_data)

#Xtr,Xva,Ytr,Yva = ml.splitData(x_data,y_data, 0.001)
Xtr = x_data
Ytr = y_data
scaler = StandardScaler()
# Fit only to the training data

scaler.fit(Xtr)

Xtr = scaler.transform(Xtr)




# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=0)


clf = svm.SVC(probability=True)
print('Training SVM Started....')
clf.fit(Xtr,Ytr)
print('Training SVM Ended....')

Ytrpred = clf.predict(Xtr)
score = clf.score(Xtr, Ytr)
errortr = np.mean(Ytr != Ytrpred)
print("Error training : " + str(errortr))
print("Score training : " + str(score))


Xte = np.genfromtxt("/data/users/anuragm/X_test.txt",delimiter=None)
Xte = Xte[:, 0:13]
Xte = scaler.transform(Xte)
Yte = clf.predict_proba(Xte)[:, 1];

fh = open('/data/users/anuragm/predictions_svm.csv','w') # open file for upload
fh.write('ID,Prob1\n') # output header line
for i,yi in enumerate(Yte):
    fh.write('{},{}\n'.format(i,yi)) # output each prediction
fh.close() # close the file

joblib.dump(clf, '/data/users/anuragm/svm_clf.pkl')
