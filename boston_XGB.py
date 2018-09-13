from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

boston = load_boston()
#print(boston.keys())
data=pd.DataFrame(boston.data)
data.columns=boston.feature_names
data.head()

X, y = data.iloc[:,:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xgb_reg = xgb.XGBRegressor()
parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.015], #so called `eta` value
              'max_depth': [2,3,4,5,6,7],
              'min_child_weight': [12, 13, 14, 15],
              'silent': [1],
              'subsample': [0.6, 0.7, 0.8, 0.9],
              'colsample_bytree': [0.7],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_reg, parameters, n_jobs=32, 
                   cv=KFold(n_splits=5),
                   scoring='mean_squared_error',
                   verbose=2, refit=True)

clf.fit(X_train, y_train)

print(clf.best_score_)
print(clf.best_estimator_)


