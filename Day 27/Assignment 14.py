import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB as g
from sklearn.metrics import accuracy_score as acc
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix as con
from  sklearn import  neighbors 
lis={}
ds=pd.read_csv("train.csv")
ds1=ds.drop(["Name","PassengerId","Parch","Cabin","Ticket"],axis=1)
from sklearn import preprocessing as pp
lab=pp.LabelEncoder()
ds1["Sex"]=lab.fit_transform(ds1["Sex"])
ds1["Embarked"]=lab.fit_transform(ds1["Embarked"])
ds1["Fare"]=ds1["Fare"].round()
ds1["Age"]=ds1["Age"].round()
def knear(n):
    y=ds1["Survived"]
    x=ds1.drop("Survived",axis=1)
    Xtrain,Xtest,Ytrain, Ytest=tts(x,y,test_size=0.3,random_state=0)
    knn=neighbors.KNeighborsClassifier(n_neighbors=n)
    acc=knn.fit(Xtrain,Ytrain,).score(Xtest,Ytest)
    ypred=knn.predict(Xtest)
    mat=con(Ytest,ypred)
    print("The value of n: ",n)
    print("The accuracy score is: ",acc)
    print("the confusion matrix is : ")
    print(mat)
for i in range(1,268):
    knear(i)
