runfile('D:/programs/WinPython-64bit-3.4.4.5Qt5/python_wrkspc/svm_grid_search.py', wdir='D:/programs/WinPython-64bit-3.4.4.5Qt5/python_wrkspc')
# Tuning hyper-parameters for precision

Best parameters set found on development set:

{'gamma': 0.001, 'kernel': 'rbf', 'C': 1000}

Grid scores on development set:

0.316 (+/-0.000) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 1}
0.316 (+/-0.000) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 1}
0.670 (+/-0.026) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 10}
0.316 (+/-0.000) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 10}
0.681 (+/-0.010) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 100}
0.677 (+/-0.027) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 100}
0.687 (+/-0.017) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 1000}
0.671 (+/-0.025) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 1000}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

        0.0       0.68      0.94      0.79    113899
        1.0       0.68      0.24      0.36     66101

avg / total       0.68      0.68      0.63    180000


# Tuning hyper-parameters for recall

Best parameters set found on development set:

{'gamma': 0.001, 'kernel': 'rbf', 'C': 1000}

Grid scores on development set:

0.500 (+/-0.000) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 1}
0.500 (+/-0.000) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 1}
0.546 (+/-0.007) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 10}
0.500 (+/-0.000) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 10}
0.575 (+/-0.008) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 100}
0.535 (+/-0.013) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 100}
0.589 (+/-0.011) for {'gamma': 0.001, 'kernel': 'rbf', 'C': 1000}
0.549 (+/-0.007) for {'gamma': 0.0001, 'kernel': 'rbf', 'C': 1000}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

        0.0       0.68      0.94      0.79    113899
        1.0       0.68      0.24      0.36     66101

avg / total       0.68      0.68      0.63    180000


