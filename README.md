loan-default-prediction
=======================

Description
-----------
The code was written for [Loan Default Prediction Competition at Kaggle](https://www.kaggle.com/c/loan-default-prediction) and got the prize. 

Dependencies and requirements
-----------------------------
[pandas](https://github.com/pydata/pandas):  version 0.13.1 or later

[scikit learn](https://github.com/scikit-learn/scikit-learn): dev branch with version commit 884889a4cd36e63d53a067d9380dea7724a93ac5 or later

How to run
----------
1. Download data from [Kaggle](https://www.kaggle.com/c/loan-default-prediction)
2. Unzip the train and test csv files to path/to/data/folder and make sure that their names are train_v2.csv and test_v2.csv, respectively
3. Run `python train_predict.py path/to/data/folder`
4. The prediction submission-ready csv (submission.csv) will be found at path/to/data/folder


