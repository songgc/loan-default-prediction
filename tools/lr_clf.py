import pandas as pd
import numpy as np
from functools import partial
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)

train_path = '\path\to\train\set'
data = read_csv(train_path)
features = ['f528-f527', 'f528-f274', 'f527-f274', 'hasnull', 'f271', 'f2',
               'f332', 'f776','f336', 'f777', 'f4', 'f5', 'f647']

train_loss = data.loss

data['hasnull'] = data.loss * 0
data['hasnull'][pd.isnull(data).any(axis=1)] = 1
data['f528-f527'] = data['f528'] - data['f527']
data['f527-f274'] = data['f527'] - data['f274']
data['f528-f274'] = data['f528'] - data['f274']
data['f272-f271'] = data['f272'] - data['f271']

train = data[features]
train = train.fillna(0)
scaler = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split( train, train_loss, test_size=0.2)

X_train = scaler.fit_transform(X_train)
y_train = np.array([1 if e > 0 else 0 for e in y_train])
y_test_v = y_test
y_test = np.array([1 if e > 0 else 0 for e in y_test])

X_test = scaler.transform(X_test)

clf = LogisticRegression(C=1e20, penalty='l2', tol=1e-6, class_weight={0: 0.2, 1: 0.8})
clf.fit(X_train,y_train)
print list(clf.coef_[0])

probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


