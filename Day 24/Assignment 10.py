import pandas as pd
import  numpy as np
from sklearn import  tree 
from sklearn import preprocessing as pp
ds=pd.read_csv("train.csv")
lan_enc=pp.LabelEncoder()
ds["Sex"]=lan_enc.fit_transform(ds["Sex"])
tm=tree.DecisionTreeClassifier(max_depth=4)
var=pd.DataFrame([ds["Sex"],ds["Age"],ds["Fare"]]).T
tree1=tm.fit(X=var,y=ds["Survived"])
with open("dtree1.dot" ,'w') as f:
    f=t.export_graphviz(tree1,feature_names=["Sex","Age","Fare"],out_file=f);
dst=pd.read_csv("test.csv")
dst["Sex"]=lan_enc.fit_transform(dst["Sex"])
test_var=pd.DataFrame([dst["Sex"],dst["Age"],dst["Fare"]]).T
test_preds=tm.predict(X=test_var)
pred_output=pd.DataFrame({"PassengerID":dst["PassengerId"],"Survied":test_preds})
pred_output.to_csv("output.csv",index=False)
