##导入数据相关的包
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

#导入数据
train=pd.read_csv("/home/mocas/kaggle/titanic/train.csv")
test=pd.read_csv("/home/mocas/kaggle/titanic/test.csv")

##总体预览数据
# print(train.head(10))
# train.info()##显示每列列名和该列的类型
# test.info()
train.describe() ##对每列数据进行处理描述，求平均最大值等
train["Survived"].value_counts()

##求相关性协方差矩阵
train_corr=train.drop('PassengerId',axis=1).corr()

##绘图
#fig=plt.figure()
# a=plt.subplot()
fig,ax = plt.subplots(figsize = (9,9)) ##创建画布,和图形对象ax
ax=sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图

##进一步观察数据与结果的关系，利用相关性分析
pclass_relate=train.groupby(['Pclass'])['Pclass','Survived'].mean()
# print(group_relate)
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot(kind='bar') ##等效于data.plot.bar(),柱状图
##分析性别和救援成功的关系
sex_relate=train.groupby(['Sex'])['Sex','Survived'].mean()

train.groupby(['Sex'])['Sex','Survived'].mean().plot.bar()
#兄妹配偶数和父母子女数
sibsp_relate=train[['SibSp','Survived']].groupby(['SibSp']).mean()
train[['Parch','Survived']].groupby(['Parch']).mean()


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# age_survived=train.Age[train.Survived==0].value_counts()
# # age_survived2=train.Survived[train.age]
# bins=np.arange(0,80,20)
# plt.hist(age_survived,bins)
# plt.show()
# # train.Age[train.Pclass == 1].plot(kind='kde')
# # n = 10
# # X = np.arange(n)+1 #X是1,2,3,4,5,6,7,8,柱的个数
# train.Age[train.Survived==0].value_counts().plot.bar()
# # plt.bar(X,train.Age[train.Survived==1].value_counts(),facecolor='#ff9999', edgecolor='white')
# plt.show()
fig=plt.figure()
sns.countplot('Embarked',hue='Survived',data=train)

##特征工程
test['Survived'] = 0
train_test = train.append(test)
print(train_test.shape)
# test.info()
train_test = pd.get_dummies(train_test,columns=['Pclass'])##作分列处理
##无缺失值直接分列
train_test = pd.get_dummies(train_test,columns=["Sex"])
##有相关性的两个特征，进行相加处理
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
train_test = pd.get_dummies(train_test,columns = ['SibSp','Parch','SibSp_Parch'])
train_test = pd.get_dummies(train_test,columns=["Embarked"])

train_test.loc[train_test["Fare"].isnull()] ##series.loc是在index寻找

#票价与pclass和Embarked有关,所以用train分组后的平均数填充
train.groupby(by=["Pclass","Embarked"]).Fare.mean()
#用pclass=3和Embarked=S的平均数14.644083来填充
train_test["Fare"].fillna(14.435422,inplace=True)

##处理age缺失数据
train_test.loc[train_test["Age"].isnull()]['Survived'].mean()##查看确实年龄的人的死亡率
# 所以用年龄是否缺失值来构造新特征
train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0
# train_test.info()
train_test = pd.get_dummies(train_test,columns=['age_nan'])

missing_age = train_test.drop(['Survived','Cabin','Ticket','Name'],axis=1)
#将Age完整的项作为训练集、将Age缺失的项作为测试集。
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
#构建训练集合预测集的X和Y值
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

# 先将数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#用测试集训练并标准化
ss.fit(missing_age_X_train)
missing_age_X_train = ss.transform(missing_age_X_train)
missing_age_X_test = ss.transform(missing_age_X_test)

#使用贝叶斯预测年龄
from sklearn import linear_model
lin = linear_model.BayesianRidge()
lin.fit(missing_age_X_train,missing_age_Y_train)
lin.predict(missing_age_X_test)
print("------这是结尾--------")

#利用loc将预测值填入数据集
train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)
train_test.drop('Cabin',axis=1,inplace=True)

#划分数据集
train_data = train_test[:891]
test_data = train_test[891:]

train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)

##标准化
from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
train_data_X=train_data_X.drop(['Name','Ticket'],axis=1)
ss2.fit(train_data_X)
train_data_X_sd = ss2.transform(train_data_X)
test_data_X_sd = ss2.transform(test_data_X)

##开始建立模型，随机森林预测
from sklearn.ensemble import RandomForestClassifier ##
rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=2,max_depth=6,oob_score=True)
rf.fit(train_data_X,train_data_Y)
rf.oob_score_

test_data_X=test_data_X.drop(['Name','Ticket'],axis=1)
test["Survived"] = rf.predict(test_data_X)

RF = test[['PassengerId','Survived']].set_index('PassengerId')
RF.info()
RF.to_csv('RF1.csv')
