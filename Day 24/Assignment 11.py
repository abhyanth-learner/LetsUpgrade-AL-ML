import pandas as pd
import  numpy as np
from sklearn import  tree as t

from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier as rf
lab=pp.LabelEncoder()
df=pd.read_csv("general_data.csv")
df["Attrition"]=lab.fit_transform(df["Attrition"])
df["BusinessTravel"]=lab.fit_transform(df["BusinessTravel"])
df["Department"]=lab.fit_transform(df["Department"])
df["EducationField"]=lab.fit_transform(df["EducationField"])
df["Gender"]=lab.fit_transform(df["Gender"])
df["JobRole"]=lab.fit_transform(df["JobRole"])
df["MaritalStatus"]=lab.fit_transform(df["MaritalStatus"])
df["Over18"]=lab.fit_transform(df["Over18"])
r_modal=rf(n_estimators=1000,max_features=2,oob_score=True)
features=['Age', 'BusinessTravel', 'Department', 'DistanceFromHome',
        'Education', 'EducationField',   'Gender',
        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
        'Over18', 'PercentSalaryHike', 'StandardHours',
        'StockOptionLevel',  'TrainingTimesLastYear',
        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
ans=r_modal.fit(X=df[features],y=df["Attrition"])
print(ans.oob_score_)
for feature,imp in zip(features, ans.feature_importances_):
    print(feature,imp)
fea=pd.DataFrame([df["Age"],df["MonthlyIncome"],df["DistanceFromHome"],df["YearsAtCompany"]]).T
tree_model=t.DecisionTreeClassifier(max_depth=4)
tree_model.fit(X=fea,y=df["Attrition"])
with open("Attrition.dot" ,'w') as f:
    f=t.export_graphviz(tree_model,feature_names=["Age","MonthlyIncome","DistanceFromHome","YearsAtCompany"],out_file=f);