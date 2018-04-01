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
from sklearn import linear_model
from sklearn.svm import SVR

data = pd.read_csv("data/training_data_all.csv")
#print(data.head())
feature_cols = ["Par-Time","Total-time","Alpha","Util"]
label_cols = ["S2","S4","S8","S12","S16","S20","S24"]


def error_rate(real,est):
    return np.mean(abs((est-real)/real*100))



#True values of kmeans
#Y_true = np.array([[1.4272409779,2.171833481,2.6365591398,2.7274749722,3.0346534653,3.188556567,3.2519893899]])
#True values of Particle Filter
Y_true = np.array([[1.3448236762, 2.3305168009, 4.2206405694, 5.7628765792, 7.1878787879, 8.7851851852, 9.1937984496]])
print(Y_true)


#X_test = [22.35,24.52,0.0884991843,0.675]  #test for kmeans kernel
X_test = [11.76,11.86,0.0084317032,0.345]   #test for Particle Filter
print(X_test)

X = data.loc[:,feature_cols]
Y = data.loc[:,label_cols]
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
Y_pred = lr.predict([X_test,])
print("Linear Regression")
print(Y_pred)
print("Error: ",mean_absolute_error(Y_true,Y_pred))
print("Error Rate: ",error_rate(Y_true,Y_pred))

temp = Y.as_matrix()

svr = SVR(C=0.1, epsilon=0.001)
svr.fit(X, temp[:,0])
Y_pred_svr = svr.predict([X_test,])
print("SVR: ")
print(Y_pred_svr)
#print("Error: ",mean_absolute_error(Y_true, Y_pred_svr))
#print("Error Rate: ",error_rate(Y_true, Y_pred_svr))

sgd = linear_model.SGDRegressor(alpha=0.000001, penalty="l2", max_iter=10000)
sgd.fit(X,temp[:,0])
Y_pred_sgd = sgd.predict([X_test,])
print("SGD: ")
print(Y_pred_sgd)
#print("Error: ",mean_absolute_error(Y_true, Y_pred_sgd))
#print("Error Rate: ",error_rate(Y_true, Y_pred_sgd))

rr = Ridge(alpha=7000)
rr.fit(X,Y)
Y_pred_ridge = rr.predict([X_test,])
print("Ridge Regression: ")
print(Y_pred_ridge)
print("Error: ",mean_absolute_error(Y_true, Y_pred_ridge))
print("Error Rate: ",error_rate(Y_true, Y_pred_ridge))
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
scores_lr = cross_val_score(lr, X, Y, cv=9)
scores_rr = cross_val_score(rr, X, Y, cv=9)
#plt.scatter(Y, scores_lr)


rf = RandomForestRegressor()
rf.fit(X, Y)
Y_pred_rf = rf.predict([X_test, ])
print("Random Forest")
print(Y_pred_rf)
print("Error: ",mean_absolute_error(Y_true,Y_pred_rf))
print("Error Rate: ",error_rate(Y_true,Y_pred_rf))


las = linear_model.Lasso(alpha=5)
las.fit(X,Y)
Y_pred_lasso = las.predict([X_test,])
print("Lasso Regression: ")
print(Y_pred_lasso)
print("Error: ",mean_absolute_error(Y_true, Y_pred_lasso))
print("Error Rate: ",error_rate(Y_true, Y_pred_lasso))




en = linear_model.ElasticNet(alpha=10)
en.fit(X,Y)
Y_pred_en = en.predict([X_test,])
print("Elastic Net: ")
print(Y_pred_en)
print("Error: ",mean_absolute_error(Y_true, Y_pred_en))
print("Error Rate: ",error_rate(Y_true, Y_pred_en))



print "----------------------------"
"""
print scores_lr
print scores_rr

print("Accuracy of LR: %0.2f (+/- %0.2f)" % (scores_lr.mean()*100, scores_lr.std() * 2))
print("Accuracy of RR: %0.2f (+/- %0.2f)" % (scores_rr.mean()*100, scores_rr.std() * 2))
"""
