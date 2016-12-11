# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:23:57 2016

@author: anurag
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
#from sklearn.preprocessing import MinMaxScaler
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
test_size = 0.10
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_2K, Y_2K, test_size=test_size, random_state=seed)


# create model
model = Sequential()
#model.add(Dropout(0.2, input_shape=(13,)))
model.add(Dense(100,input_dim=13, init='lecun_uniform', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(50, init='lecun_uniform', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(25, init='lecun_uniform', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, init='lecun_uniform', activation='sigmoid'))


# load weights
model.load_weights("weights.best.hdf5")

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Created model and loaded weights from file")



# evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

X_kaggle=numpy.loadtxt("data/X_test.txt", delimiter=" ")
X_kaggle=X_kaggle[:,:13]
X_kaggle = scaler.transform(X_kaggle)
# calculate predictions
predictions = model.predict(X_kaggle)

fh = open('data/predictions.csv','w') # open file for upload
fh.write('ID,Prob1\n') # output header line
for i,yi in enumerate(predictions):
    fh.write('{},{}\n'.format(i,predictions[i][0])) # output each prediction
fh.close() # close the file

from sklearn.metrics import confusion_matrix
results = confusion_matrix(Y_test, numpy.around(model.predict(X_test) , decimals=0))
print(results)