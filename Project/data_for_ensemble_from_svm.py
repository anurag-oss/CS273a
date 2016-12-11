# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:10:01 2016

@author: anurag
"""


import numpy

from sklearn.preprocessing import StandardScaler


from sklearn import svm
from sklearn.externals import joblib
#from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

seed = 7
numpy.random.seed(seed)

X_2K=numpy.loadtxt("data/X_train.txt", delimiter=" ")
X_2K=X_2K[:,:13]
Y_2K=numpy.loadtxt("data/Y_train.txt")
Y_2K.shape = (Y_2K.shape[0],1)

scaler = StandardScaler().fit(X_2K)
#scaler = MinMaxScaler().fit(X_2K)
X_2K = scaler.transform(X_2K)

print(X_2K.shape)
print(Y_2K.shape)



clf = joblib.load('svm_clf.pkl')
print('Loaded the weights')


predictions = clf.predict(X_2K)
print('Predictions done')

fh = open('stage1_output_from_svm.csv','w') 

for i,yi in enumerate(predictions):
    fh.write('{}\n'.format(predictions[i])) # output each prediction
fh.close() # close the file