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

adultdata=pd.read_csv('adult.train.txt',sep=', ',names = ['Age', 'Workclass', 'Fnlwgt', 'Education','Education num','Marital Status',
                                          'Occupation','Relationship','Race','Sex','Capital Gain','capital loss',
                                          'Hour per week',
                                          'Native country','income'])

adultdatatest=pd.read_csv('adult.test.txt',sep=', ',names = ['Age', 'Workclass', 'Fnlwgt', 'Education','Education num','Marital Status',
                                          'Occupation','Relationship','Race','Sex','Capital Gain','capital loss',
                                          'Hour per week',
                                          'Native country','income'])
#adultdataframr=pd.DataFrame(adultdata)
print(adultdata.keys()) 
print(adultdata.shape) #計算(32561, 15) 32651列，15行 的 陣列
print("===========================================")
print(adultdata)
print("===========================================")

#print(adultdata.head()) #顯示前幾筆資料
#print(adultdata.info())

#print(adultdata[adultdata['Workclass']=='?'])
#print(adultdata.nunique()) nunique() 方法用于获取 'Team’列中所有唯一值的数量

print(adultdata.describe())
print(adultdata.columns)


print("===========================================")
#sns.countplot(adultdata['income'],palette='coolwarm',hue='relationship',data=adultdata)

print("===========================================")
#print(adultdata['Workclass'].value_counts())
print("===========================================")

adultdata['Workclass']=adultdata['Workclass'].replace("?","Private")
adultdata['Occupation']=adultdata['Occupation'].replace("?","Prof-specialty")
adultdata['Native country']=adultdata['Native country'].replace("?","United-States")

#adultdata.to_csv("s4.txt")檔案輸出

# education Category
adultdata.Education= adultdata.Education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'school')
adultdata.Education = adultdata.Education.replace('HS-grad', 'high school')
adultdata.Education = adultdata.Education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'higher')
adultdata.Education = adultdata.Education.replace('Bachelors', 'undergrad')
adultdata.Education = adultdata.Education.replace('Masters', 'grad')
adultdata.Education = adultdata.Education.replace('Doctorate', 'doc')

#martial status
adultdata['Marital Status']= adultdata['Marital Status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'married')
adultdata['Marital Status']= adultdata['Marital Status'].replace(['Never-married'], 'not-married')
adultdata['Marital Status']= adultdata['Marital Status'].replace(['Divorced', 'Separated','Widowed',
                                                   'Married-spouse-absent'], 'other')

# income
#adultdata.income = adultdata.income.replace('<=50K', 0)
#adultdata.income = adultdata.income.replace('>50K', 1)
adultdata.income = adultdata.income.replace('> 0K', '<=50K')

print(adultdata['income'].value_counts())
print("===========================================")

adultdatatest['Workclass']=adultdatatest['Workclass'].replace("?","Private")
adultdatatest['Occupation']=adultdatatest['Occupation'].replace("?","Prof-specialty")
adultdatatest['Native country']=adultdatatest['Native country'].replace("?","United-States")

#adultdata.to_csv("s4.txt")檔案輸出

# education Category
adultdatatest.Education= adultdatatest.Education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'school')
adultdatatest.Education = adultdatatest.Education.replace('HS-grad', 'high school')
adultdatatest.Education = adultdatatest.Education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'higher')
adultdatatest.Education = adultdatatest.Education.replace('Bachelors', 'undergrad')
adultdatatest.Education = adultdatatest.Education.replace('Masters', 'grad')
adultdatatest.Education = adultdatatest.Education.replace('Doctorate', 'doc')

#martial status
adultdatatest['Marital Status']= adultdatatest['Marital Status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'married')
adultdatatest['Marital Status']= adultdatatest['Marital Status'].replace(['Never-married'], 'not-married')
adultdatatest['Marital Status']= adultdatatest['Marital Status'].replace(['Divorced', 'Separated','Widowed',
                                                   'Married-spouse-absent'], 'other')

# income

adultdatatest.income = adultdatatest.income.replace('> 0K', '<=50K')
print("===========================================")
#data.corr()表示了data中的两个变量之间的相关性，取值范围为[-1,1],取值接近-1，表示反相关，类似反比例函数，取值接近1，表正相关。
#print(adultdata.corr())
#sns.heatmap(adultdata.corr(),annot=True)

#label=adultdata['income']
#for i in range(len(label)):
    #if(label[i] == '<=50K'):
        #label[i]=0
    #elif(label[i] == '>=50K'):
        #label[i]=1
    #elif(label[i] == '> 0K'):
        #label[i]=2
print("=========================================== 將屬性數值化")
#Label encoding : 把每個類別 mapping 到某個整數，不會增加新欄位
#One hot encoding : 為每個類別新增一個欄位，用 0/1 表示是否
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
le=LabelEncoder()
adultdata1=adultdata.copy()
#adultdata1=adultdata1.apply(LabelEncoder().fit_transform)
#for col in adultdata1.columns:
    
    #if adultdata1[col].dtypes=='object':
        #adultdata1[col]=le.fit_transform(adultdata1[col])

adultdata1['Workclass'] = le.fit_transform(adultdata['Workclass'])
adultdata1['Education'] = le.fit_transform(adultdata['Education'])
adultdata1['Marital Status'] = le.fit_transform(adultdata['Marital Status'])
adultdata1['Occupation'] = le.fit_transform(adultdata['Occupation'])
adultdata1['Relationship'] = le.fit_transform(adultdata['Relationship'])
adultdata1['Race'] = le.fit_transform(adultdata['Race'])
adultdata1['Sex'] = le.fit_transform(adultdata['Sex'])
adultdata1['Native country'] = le.fit_transform(adultdata['Native country'])
adultdata1['income'] = le.fit_transform(adultdata['income'])
#adultdata1['income']=pd.to_numeric(adultdata1['income'])
print("info :")
print(adultdata1.info())
#onehotencoder=OneHotEncoder(Category_features=[0])
#adultdata12=pd.get_dummies(adultdata)
print("==============================================")
adultdatatest1=adultdatatest.copy()
adultdatatest1['Workclass'] = le.fit_transform(adultdatatest['Workclass'])
adultdatatest1['Education'] = le.fit_transform(adultdatatest['Education'])
adultdatatest1['Marital Status'] = le.fit_transform(adultdatatest['Marital Status'])
adultdatatest1['Occupation'] = le.fit_transform(adultdatatest['Occupation'])
adultdatatest1['Relationship'] = le.fit_transform(adultdatatest['Relationship'])
adultdatatest1['Race'] = le.fit_transform(adultdatatest['Race'])
adultdatatest1['Sex'] = le.fit_transform(adultdatatest['Sex'])
adultdatatest1['Native country'] = le.fit_transform(adultdatatest['Native country'])
adultdatatest1['income'] = le.fit_transform(adultdatatest['income'])




print("===========================================")

#Standardization 平均&變異數標準化
#將所有特徵標準化，也就是高斯分佈。使得數據的平均值為0，方差為1。
#適合的使用時機於當有些特徵的方差過大時，使用標準化能夠有效地讓模型快速收斂
#std=StandardScaler().fit(adultdata1.drop('Hour per week',axis=1))
#X=std.transform(adultdata1.drop('Hour per week',axis=1))

scaler=StandardScaler()
X_train=scaler.fit_transform(adultdata1.drop('Hour per week',axis=1))
#print("X :")
#print(X[7])
print("X_train :")
print(X_train)
X_validation=adultdata1['Hour per week'].values.reshape(-1,1)


y_test=scaler.fit_transform(adultdatatest1.drop('Hour per week',axis=1))
#print("X :")
#print(X[7])

y_validation=adultdatatest1['Hour per week'].values.reshape(-1,1)
print(adultdata1)
adultdata1.to_csv("adulttrainpre.csv")


#X=adultdata1.copy()




#X=adultdata1.drop('income',axis=1)
#X=adultdata1.drop('Hour per week',axis=1)
#X=adultdata1[['Age','Fnlwgt','Education num','Occupation','Capital Gain','capital loss','income']]
#y=adultdata1['Hour per week']



from sklearn.model_selection import train_test_split
#X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=0.2,random_state=0)
#print("x:")
#print(X_validation)
#print("y:")
#print(y_validation)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
## 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#lr=LogisticRegression()
#model=lr.fit(X_train,y_train)
#prediction=model.predict(X_validation)

#print("LogisticRegression")
#print("Acc on training data:  {:,.3f}".format(lr.score(X_train,y_train)))
#print(prediction)
#print(y_validation)
#print("MAE :")
#print(mean_absolute_error(y_validation,prediction))
#print("MAPE :")
#print(mape(y_validation,prediction))
#print("RMSE")
#print(np.sqrt(metrics.mean_squared_error(y_validation,prediction)))
print("===========================================")
from sklearn import linear_model
from sklearn.datasets import make_regression
lr=linear_model.LinearRegression()
lr.fit(X_train,X_validation)

## 使用訓練集資料來訓練(擬和)迴歸模型，多項式
#regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())
#regressor.fit(X_train, y_train)
y_pred = lr.predict(y_test)
print("LinearRegression")
print("MAE :")
print(mean_absolute_error(y_validation,y_pred))
print("MAPE :")
print(mape(y_validation,y_pred))
print("RMSE")
print(np.sqrt(metrics.mean_squared_error(y_validation,y_pred)))
print("Acc on training data:  {:,.3f}".format(lr.score(X_train,X_validation)))
print("Acc on test data: {:,.3f}".format(lr.score(y_test, y_validation)))
print("===========================================")

#random forest classifier
#rfc=RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=(0),n_jobs=2,max_depth=(14))

#model1 =rfc.fit(X_train,y_train)
#prediction1=model1.predict(X_validation) 
#print("RandomForestClassifier")
#print("Acc on training data: {:,.3f}".format(rfc.score(X_train, y_train)))
#print("Acc on test data: {:,.3f}".format(rfc.score(X_validation, y_validation)))
#print("MAE :")
#print(mean_absolute_error(y_validation,prediction1))
#print("MAPE :")
#print(mape(y_validation,prediction1))
#print("RMSE")
#print(np.sqrt(metrics.mean_squared_error(y_validation,prediction1)))
#txt1=pd.DataFrame(y_validation)
#txt1.to_csv('randomforestreal.txt')
#print(txt1.value_counts())
#txt=pd.DataFrame(prediction1)
#print(txt.value_counts())
#txt.to_csv('randomforest.txt')

#random forest Regression
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,oob_score=(True),n_jobs=(1),random_state=(1))
 
rfr.fit(X_train,X_validation)   #训练数据
prediction1=rfr.predict(y_test)
print("RandomForestregression")
print("Acc on training data: {:,.3f}".format(rfr.score(X_train, X_validation)))
print("Acc on test data: {:,.3f}".format(rfr.score(y_test, y_validation)))
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#混淆矩陣
#print(confusion_matrix(y_validation,prediction1))
#precision    recall  f1-score   support 
#print(classification_report(y_validation,prediction1))



print("===========================================")
#knn 
from sklearn.neighbors import KNeighborsClassifier
#knn=KNeighborsClassifier(n_neighbors=10)
#knn.fit(X_train,y_train)
#predict2=knn.predict(X_validation)
#print(y_validation)
#print(confusion_matrix(y_validation,predict2))
#print((6872+746)/(6872+1568+583+746))
#print(classification_report(y_validation,predict2))
#print("knn :")
#print("MAE :")
#print(mean_absolute_error(y_validation,predict2))
#print("MAPE :")
#print(mape(y_validation,predict2))
#print("RMSE")
#print(np.sqrt(metrics.mean_squared_error(y_validation,predict2)))
#preict2.to_csv



#knn回歸器
from sklearn.neighbors import KNeighborsRegressor
#建立knn模型
knnmodel=KNeighborsRegressor(n_neighbors=101,weights='distance',p=2)
# 使用訓練資料訓練模型
knnmodel.fit(X_train,X_validation)
## 使用訓練資料預測
predict=knnmodel.predict(y_test)

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
print("Acc on training data: {:,.3f}".format(knnmodel.score(X_train, X_validation)))
print("Acc on test data: {:,.3f}".format(knnmodel.score(y_test, y_validation)))


print("===========================================")
print("XGBoost")
#XGBoost (迴歸器)
#import xgboost as xgb
#from xgboost import XGBClassifier
from xgboost.sklearn import XGBRegressor
# 建立 XGBRegressor 模型
xgbrModel=XGBRegressor()
# 使用訓練資料訓練模型
xgbrModel.fit(X_train,X_validation)
# 使用訓練資料預測
predicted=xgbrModel.predict(y_test)

# 預測成功的比例
print('訓練集: ',xgbrModel.score(X_train,X_validation))
print('測試集: ',xgbrModel.score(y_test,y_validation))
#特徵重要性
impxgb=xgbrModel.feature_importances_
print("特徵重要性:")
importfeaturesxgr=pd.DataFrame(impxgb,columns=['featureimportant'])
print(importfeaturesxgr)
