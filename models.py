'''
@author: guocong
'''
import numpy as np
import pandas as pd
from collections import defaultdict


class FeatureSelector(object):

    def __init__(self):
        self.strong = ['f528-f527', 'f528-f274', 'f527-f274', 'hasnull', 'f271', 'f2',
                       'f332', 'f776','f336', 'f777', 'f4', 'f5', 'f647', 'trial2']
        self.tail_feature_300 = ['f607', 'f723', 'f453', 'f452', 'f604', 'f454', 'f457', 'f456', 'f459', 'f163', 'f162', 'f164', 'f752', 'f769', 'f502', 'f510', 'f455', 'f512', 'f762', 'f761', 'f283', 'f688', 'f206', 'f42', 'f750', 'f606', 'f681', 'f683', 'f684', 'f685', 'f686', 'f687', 'f608', 'f505', 'f504', 'f469', 'f610', 'f549', 'f616', 'f615', 'f507', 'f758', 'f759', 'f420', 'f119', 'f299', 'f503', 'f115', 'f501', 'f500', 'f293', 'f757', 'f291', 'f113', 'f215', 'f214', 'f511', 'f192', 'f692', 'f691', 'f690', 'f697', 'f506', 'f592', 'f595', 'f550', 'f537', 'f534', 'f94', 'f530', 'f531', 'f198', 'f625', 'f749', 'f748', 'f538', 'f539', 'f644', 'f435', 'f345', 'f109', 'f346', 'f472', 'f105', 'f104', 'f103', 'f439', 'hasnull', 'f95', 'f184', 'f182', 'f224', 'f497', 'f188', 'f583', 'f581', 'f580', 'f585', 'f584', 'f521', 'f28', 'f633', 'f730', 'f736', 'f482', 'f483', 'f480', 'f481', 'f486', 'f487', 'f484', 'f485', 'f400', 'f207', 'f285', 'f323', 'f130', 'f131', 'f515', 'f137', 'f527', 'f335', 'f232', 'f33', 'f36', 'f48', 'f678', 'f568', 'f125', 'f722', 'f720', 'f120', 'f123', 'f724', 'f728', 'f129', 'f747', 'f554', 'f555', 'f557', 'f419', 'f551', 'f552', 'f415', 'f414', 'f558', 'f559', 'f124', 'f753', 'f325', 'f89', 'f321', 'f85', 'f84', 'f242', 'f83', 'f190', 'f675', 'f627', 'f528', 'f770', 'f152', 'f718', 'f719', 'f158', 'f389', 'f714', 'f712', 'f713', 'f710', 'f711', 'f547', 'f546', 'f90', 'f544', 'f543', 'f542', 'f541', 'f540', 'f460', 'f461', 'f99', 'f464', 'f466', 'f548', 'f317', 'f315', 'f252', 'f495', 'f494', 'f18', 'f496', 'f491', 'f490', 'f493', 'f492', 'f576', 'f709', 'f708', 'f387', 'f329', 'f347', 'f8', 'f703', 'f705', 'f704', 'f707', 'f706', 'f572', 'f570', 'f571', 'f262', 'f577', 'f574', 'f575', 'f114', 'f578', 'f579', 'f477', 'f476', 'f268', 'f470', 'f309', 'f666', 'f307', 'f301', 'f475', 'f78', 'f62', 'f396', 'f446', 'f447', 'f445', 'f174', 'f172', 'f173', 'f449', 'f569', 'f274', 'f565', 'f564', 'f567', 'f777', 'f561', 'f560', 'f772', 'f562', 'f52', 'f467', 'f93', 'f372', 'f355', 'f284', 'f754', 'f77']
        self.exclude = self.tail_feature_300
        
    def _reduce(self, df, columns):
        table = defaultdict(list)
        for col in columns:
            vec = df[col]
            key = (vec.dropna().count(), vec.min(), vec.max(), vec.mean(), vec.median())
            table[key].append(col)
        return [a[0] for a in table.values()]
        
    def fit(self, X, y=None):
        dtypes = X.dtypes.apply(lambda x: x.name).to_dict()
        int_cols, float_cols, str_cols = [], [], []
        for col, dtype in dtypes.iteritems():
            if dtype == 'int64' and col not in ['id', 'loss']:
                int_cols.append(col)
            elif dtype == 'float64':
                float_cols.append(col)
            elif dtype == 'object':
                str_cols.append(col)
        
        int_cols.sort(), float_cols.sort(), str_cols.sort()
        int_cols = self._reduce(X, int_cols)
        float_cols = self._reduce(X, float_cols)
        self.int_cols = int_cols
        self.float_cols = float_cols
        self.str_cols = str_cols
        self._convert(X, str_cols)
        self.features = int_cols + float_cols + str_cols
        self.features = self.strong + [e for e in self.features if e not in self.exclude + self.strong]
        return self
        
    def transform(self, X):
        data = X
        data['f528-f527'] = data['f528'] - data['f527']
        data['f528-f274'] = data['f528'] - data['f274']
        data['f527-f274'] = data['f527'] - data['f274']
        data['trial2'] = 5*data['f528'] - 4*data['f274'] - data['f527']
        data['hasnull'] = np.zeros(len(data))
        data['hasnull'][pd.isnull(data).any(axis=1)] = 1
        
        cols = ['f528-f527', 'f528-f274', 'f527-f274', 'hasnull', 'f271', 'f2']
        weights = [8.9039, 2.9559, -2.6462, -0.0391, -7.3605, 1.9536]
        self._make_new_feature(data, 'comb', cols, weights)
        
        cols = ['f528-f527', 'f528-f274', 'f527-f274', 'hasnull', 'f271', 'f2', \
               'f332', 'f776','f336', 'f777', 'f4', 'f5', 'f647']
        weights = [9.9500765961934405, 3.7781687215318116, -1.8698562077221303, -0.066277197757217091, -6.6669154039560761, 2.5128446862308884, -2.1690997464054576, -0.092630191672692777, -2.4930731337790117, 0.65168453290941586, -0.12754136033274549, -0.018338717738365744, 0.013455903115007804]
        self._make_new_feature(data, 'comb1', cols, weights)
        
        cols = ['f528-f527', 'f528-f274', 'f527-f274', 'hasnull', 'f271', 'f2',
               'f332', 'f776','f336', 'f777', 'f4', 'f5', 'f647', 'f67', 'f670', 'f598', 'f596']    
        weights = [1.0634560449318777, 0.29053072516872058, -0.40554815884849466, 0.046776844565513807, -2.4241799177037078, 0.19964143173757548, -6.9566421015981108, 0.20513531048071851, -0.34675217706015837, 0.2350456207924046, -0.066067751148901038, 0.11147667983919719, -0.054502336657992208, 0.25678764279409982, 0.34284236736287044, 0.61539930323510994, -0.75889755596066966]
        self._make_new_feature(data, 'comb2', cols, weights)
        
        return data[self.features].astype(np.float).values
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)   
    
    def _convert(self, df, features):
        for fea in features:
            df[fea] = df[fea].astype(float)
            
    def _make_new_feature(self, data, name, features, weights):
        data[name] = 0
        for col, w in zip(features, weights):
            data[name] += data[col] * w
        if name not in self.features:
            self.features.append(name)
        
 
class DefaultClassifier(object):
    
    def __init__(self, clf):
        self.clf = clf
        
    def fit(self, X, y):
        yy = np.copy(y)
        yy[yy > 0] = 1.0
        self.clf.fit(X, yy)
        
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        
    
class PartialRegressor(object):
    
    def __init__(self, clf):
        self.clf = clf
        
    def fit(self, X, y):
        yy = y[y > 0]
        XX = X[y > 0]
        yy = np.log(yy)
        self.clf.fit(XX, yy)
        return self
    
    def predict(self, X):
        y = self.clf.predict(X)
        y = np.exp(y)
        y[y < 1] = 1.0
        y[y > 100] = 100.0
        return y         
