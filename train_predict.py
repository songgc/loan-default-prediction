'''
@author: guocong
'''
import pandas as pd
import os
import sys
from functools import partial
import models
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)


def get_clf_pipeline():
    clf = models.DefaultClassifier(
            GradientBoostingClassifier(
                       loss='deviance', learning_rate=0.01, n_estimators=3000,
                       subsample=0.6, min_samples_split=12, min_samples_leaf=12,
                       max_depth=6, random_state=1357, verbose=0)
           )
    steps = [('features', models.FeatureSelector()),
             ('Impute', Imputer(strategy='median')),
             ('scaler', StandardScaler()),
             ('clf', clf)]
    return Pipeline(steps)


def get_reg_pipeline():
    clf = models.PartialRegressor(
            GradientBoostingRegressor(loss='ls', learning_rate=0.0075, n_estimators=5000,
                 subsample=0.5, min_samples_split=20, min_samples_leaf=20, max_leaf_nodes=30,
                 random_state=9753, verbose=0)
            )
    steps = [('features', models.FeatureSelector()),
             ('Impute', Imputer(strategy='median')),
             ('scaler', StandardScaler()),
             ('clf', clf)]
    return Pipeline(steps)


def default_clf(train_file, test_file, path):
    train = read_csv(train_file)
    pipeline = get_clf_pipeline()
    pipeline.fit(train, train['loss'].values)
    
    test = read_csv(test_file)
    proba = pipeline.predict_proba(test)
    _, n = proba.shape
    columns = ['c' + str(i) for i in range(n)]
    proba = pd.DataFrame(proba, columns=columns)
    filepath = os.path.join(path, 'test_proba.csv')
    proba.to_csv(filepath, index=False)
    

def loss_reg(train_file, test_file, path):
    train = read_csv(train_file)
    pipeline = get_reg_pipeline()
    pipeline.fit(train, train['loss'].values)
    
    test = read_csv(test_file)
    prob_default_test = pd.read_csv(os.path.join(path, 'test_proba.csv'))['c1']
    pred_loss = pipeline.predict(test)
    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['loss'] = pred_loss * (prob_default_test.values > 0.6)
    submission.to_csv(os.path.join(path, 'submission.csv'), index=False)
    

if __name__ == '__main__':
    path = sys.argv[1]
    filenames = ['train_v2.csv', 'test_v2.csv']
    train_file, test_file = [os.path.join(path, fname) for fname in filenames]
    print 'training set is', train_file
    print 'test set is', test_file
    print 'default classifier training...'
    default_clf(train_file, test_file, path)
    print 'loss regression training...'
    loss_reg(train_file, test_file, path)
    