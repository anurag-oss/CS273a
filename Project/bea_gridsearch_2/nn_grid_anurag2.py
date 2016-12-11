# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:42:09 2016

@author: anurag
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

seed = 7
numpy.random.seed(seed)

X_2K=numpy.loadtxt("data/X_train.txt", delimiter=" ")
Y_2K=numpy.loadtxt("data/Y_train.txt")
Y_2K.shape = (Y_2K.shape[0],1)

scaler = StandardScaler().fit(X_2K)
X_2K = scaler.transform(X_2K)

print(X_2K.shape)
print(Y_2K.shape)
test_size = 0.10
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_2K, Y_2K, test_size=test_size, random_state=seed)


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(14, input_dim=14, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

    
# create model
model = KerasClassifier(build_fn=create_model,nb_epoch=20, validation_split=0.1, verbose=1) 
print('Model created...')   
# define the grid search parameters
batch_size = [10, 20, 30, 50]
param_grid = dict(batch_size=batch_size)
print('Starting Grid Search...')
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#joblib.dump(grid_result.best_score_ , 'data/best_score_bea.pkl')
#joblib.dump(grid_result.best_params_ , 'data/best_params_bea.pkl')


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
numpy.savetxt("data/means_bea.csv", means, delimiter=",")
numpy.savetxt("data/stds_bea.csv", stds, delimiter=",")
with open('data/params_bea.csv', 'wb') as output_file:
    for row in params:
        s=str(row['batch_size'])+'\n'
        output_file.write(bytes(s, 'UTF-8'))
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

    
#joblib.dump(means , "data/means_bea.pkl")
#joblib.dump(stds , "data/stds_bea.pkl")
#joblib.dump(params , "data/params_bea.pkl")    


    

       

