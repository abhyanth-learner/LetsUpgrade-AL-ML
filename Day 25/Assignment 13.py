import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB as g
from sklearn.metrics import accuracy_score as acc
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix as con
ds=pd.read_csv("train.csv")
ds1=ds.drop(["Name","PassengerId","Parch","Cabin","Ticket"],axis=1)
from sklearn import preprocessing as pp
lab=pp.LabelEncoder()
ds1["Sex"]=lab.fit_transform(ds1["Sex"])
ds1["Embarked"]=lab.fit_transform(ds1["Embarked"])
ds1["Fare"]=ds1["Fare"].round()
ds1["Age"]=ds1["Age"].round()
def naive (dv):
    y=ds1[dv]
    x=ds1.drop(dv,axis=1)
    print("The dependant variable is: ",dv)
    print("The INdependant variables are: ",x.columns)
    Xtrain,Xtest,Ytrain, Ytest=tts(x,y,test_size=0.3,random_state=0)
    ber=BernoulliNB()
    ypred=ber.fit(Xtrain,Ytrain).predict(Xtest)
    accur=acc(Ytest,ypred,normalize=True)
    conmat=con(Ytest,ypred)
    print("THE acciracy score is: ",accur)
    print("THE confucion matrix is: ")
    print(conmat)
for i in ds1:
    naive(i)
   
         
    