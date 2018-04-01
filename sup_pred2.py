import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import seaborn as sb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier

data = pd.read_csv("data/training_data_all.csv")
#print(data.head())
feature_cols = ["Par-Time","Total-time","Alpha","Util"]
label_cols = ["S2","S4","S8","S12","S16","S20","S24"]
cols = [0,1,2,3,4,5,6]
crv = 2

X_test = [22.35,24.52,0.0884991843,0.675]
Y_true = np.array([[1.4272409779,2.171833481,2.6365591398,2.7274749722,3.0346534653,3.188556567,3.2519893899]])
print(X_test)
print(Y_true.shape)
print(Y_true)


def error_rate(real,est):
    return abs((est-real)/real*100)


def lr_rr(col,label_col):
    X = data.loc[:,feature_cols]
    Y = data.loc[:,label_col]
    lr = LinearRegression(normalize=True)
    lr.fit(X,Y)
    Y_pred = lr.predict([X_test,])
    print("Testing for: ",Y_true[:,col])
    print("Linear Regression")
    print(Y_pred)
    print("Error: ",mean_absolute_error(Y_true[:,col],Y_pred))
    print("Error Rate: ",error_rate(Y_true[:,col],Y_pred))

    rr = Ridge(normalize=True)
    rr.fit(X,Y)
    Y_pred_ridge = rr.predict([X_test,])
    print("Ridge Regression: ")
    print(Y_pred_ridge)
    print("Error: ",mean_absolute_error(Y_true[:,col], Y_pred_ridge))
    print("Error Rate: ",error_rate(Y_true[:,col], Y_pred_ridge))
    #print(lr.score(X,Y))
    #print(rr.score(X,Y))
    """
    kf = KFold(n_splits=6) # Define the split - into 2 folds
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

    print(kf)
    for train_index, test_index in kf.split(X):
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = Y[train_index], Y[test_index]

    """
    scores_lr = cross_val_score(lr, X, Y, cv=crv)
    scores_rr = cross_val_score(rr, X, Y, cv=crv)
    #plt.scatter(Y, scores_lr)


    print scores_lr
    print scores_rr

    print("Accuracy of LR: %0.2f (+/- %0.2f)" % (scores_lr.mean()*100, scores_lr.std() * 2))
    print("Accuracy of RR: %0.2f (+/- %0.2f)" % (scores_rr.mean()*100, scores_rr.std() * 2))

for col,label_col in zip(cols,label_cols):
    lr_rr(col,label_col)

