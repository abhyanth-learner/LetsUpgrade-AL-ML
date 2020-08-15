import pandas as pd
import  numpy as np
from sklearn import  tree as t

from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier as rf
lab=pp.LabelEncoder()
df1=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)
r_modal1=rf(n_estimators=1000,max_features=2,oob_score=True)
features=['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']
ans1=r_modal1.fit(X=df1[features],y=df1["Personal Loan"])
print(ans1.oob_score_)
for feature1,imp1 in zip(features, ans1.feature_importances_):
    print(feature1,imp1)
fea=pd.DataFrame([df1["Income"],df1["Education"],df1["CCAvg"],df1["Family"]]).T
tree_model=t.DecisionTreeClassifier(max_depth=4)
tree_model.fit(X=fea,y=df1["Personal Loan"])
with open("Personal_Loan.dot" ,'w') as f:
    f=t.export_graphviz(tree_model,feature_names=["Income","Education","CCAvg","Family"],out_file=f);