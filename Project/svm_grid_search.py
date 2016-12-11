import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import mltools as ml

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

seed = 7
np.random.seed(seed)

x_data = np.genfromtxt("data/X_train.txt",delimiter=None)
y_data = np.genfromtxt("data/Y_train.txt",delimiter=None)

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
X_train, X_test, y_train, y_test = train_test_split(Xtr, Ytr, test_size=0.9, random_state=seed)


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
