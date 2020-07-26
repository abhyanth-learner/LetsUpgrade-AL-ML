import numpy as n
import pandas as pd
import matplotlib.pyplot as m
from scipy.stats import pearsonr as p, mannwhitneyu as man,ttest_ind as ttest
from  sklearn.preprocessing import LabelEncoder
count=0
ds=pd.read_csv("general_data.csv")
ds.dropna()
ds.drop_duplicates()
le=LabelEncoder()
ds["Attrition"]=le.fit_transform(ds["Attrition"])
data_yes=ds[ds["Attrition"]==1]
data_no=ds[ds["Attrition"]==0]
#=============================================================================
# s1=data_yes.DistanceFromHome
# s2=data_no.DistanceFromHome
# stat,p=man(s1,s2)
# print("The value of p is",p)
#=============================================================================
# =============================================================================
# s3=data_yes.JobLevel
# s4=data_no.JobLevel
# stat,p=man(s3,s4)
# print("The value of p is",p)
# 
# 
# =============================================================================
# =============================================================================
# =============================================================================
# s5=data_yes.YearsWithCurrManager
# s6=data_no.YearsWithCurrManager
# stat,p=man(s6,s5)
# print("The value of p is",p)
# =============================================================================
# =============================================================================
# s1=data_yes.DistanceFromHome
# s2=data_no.DistanceFromHome
# stat,p=ttest(s1,s2)
# print("The value of p is",p)
# =============================================================================
s3=data_yes.JobLevel
s4=data_no.JobLevel
stat,p=ttest(s3,s4)
print("The value of p is",p)