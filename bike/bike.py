import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error   #MAE
from sklearn import metrics

#Mape
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


#import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

hourdata=pd.read_csv('hour.csv')
print(hourdata)

print(hourdata.keys())
print(hourdata.shape)

print("info :",hourdata.info())
#print("describe :",hourdata.describe())

#for col in hourdata.columns:
    #print(hourdata[col].value_counts())

trainhourdata=hourdata.copy()
trainhourdata=trainhourdata.drop(['instant','cnt','dteday'],axis=1).values

print(trainhourdata)
testhourday=hourdata['cnt'].values.reshape(-1,1)
print("================================")
print(testhourday)

from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation=train_test_split(trainhourdata,testhourday,test_size=0.3)

























from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
## 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline






lr=LinearRegression()
lr.fit(X_train,y_train)

## 使用訓練集資料來訓練(擬和)迴歸模型，多項式
#regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())
#regressor.fit(X_train, y_train)
y_pred = lr.predict(X_validation)
print("LinearRegression")
print("MAE :")
print(mean_absolute_error(y_validation,y_pred))
print("MAPE :")
print(mape(y_validation,y_pred))
print("RMSE")
print(np.sqrt(metrics.mean_squared_error(y_validation,y_pred)))
print("Acc on training data:  {:,.3f}".format(lr.score(X_train,y_train)))
print("Acc on test data:  {:,.3f}".format(lr.score(X_validation,y_validation)))
print("===========================================")

#random forest Regression
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,oob_score=(True),n_jobs=(1),random_state=(1))
 
rfr.fit(X_train,y_train)   #训练数据
prediction1=rfr.predict(X_validation)
print("RandomForestregression")
print("Acc on training data: {:,.3f}".format(rfr.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(rfr.score(X_validation, y_validation)))
print("MAE :")
print(mean_absolute_error(y_validation,prediction1))
print("MAPE :")
print(mape(y_validation,prediction1))
print("RMSE")
print(np.sqrt(metrics.mean_squared_error(y_validation,prediction1)))

#特徵重要性
imp=rfr.feature_importances_
print("特徵重要性:")
importfeatures=pd.DataFrame(imp,columns=['featureimportant'])
print(importfeatures)


print("===========================================")

#knn回歸器
from sklearn.neighbors import KNeighborsRegressor
#建立knn模型
knnmodel=KNeighborsRegressor(n_neighbors=101,weights='distance',p=2)
# 使用訓練資料訓練模型
knnmodel.fit(X_train,y_train)
## 使用訓練資料預測
predict=knnmodel.predict(X_validation)

from sklearn import metrics
#print(knnmodel.score(X_train,y_train))
print("knn回歸器")
print("MAE :")
print(mean_absolute_error(y_validation,predict))
print("MAPE :")
print(mape(y_validation,predict))
print("RMSE")
print(np.sqrt(metrics.mean_squared_error(y_validation,predict)))


#print(predict)
#pd1=pd.DataFrame(predict)
#print(pd1)
#print("==")
#print(y_validation)
print("Acc on training data: {:,.3f}".format(knnmodel.score(X_train,y_train)))
print("Acc on test data: {:,.3f}".format(knnmodel.score(X_validation,y_validation)))

print("===========================================")
print("XGBoost")
#XGBoost (迴歸器)
#import xgboost as xgb
#from xgboost import XGBClassifier
from xgboost.sklearn import XGBRegressor
# 建立 XGBRegressor 模型
xgbrModel=XGBRegressor()
# 使用訓練資料訓練模型
xgbrModel.fit(X_train,y_train)
# 使用訓練資料預測
predicted=xgbrModel.predict(X_validation)

# 預測成功的比例
print('訓練集: ',xgbrModel.score(X_train,y_train))
print('測試集: ',xgbrModel.score(X_validation,y_validation))
#特徵重要性
impxgb=xgbrModel.feature_importances_
print("特徵重要性:")
importfeaturesxgr=pd.DataFrame(impxgb,columns=['featureimportant'])
print(importfeaturesxgr)
