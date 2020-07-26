import numpy as n
import pandas as pd
import matplotlib.pyplot as m
from scipy.stats import pearsonr as pear, mannwhitneyu as man,ttest_ind as ttest
from  sklearn.preprocessing import LabelEncoder
count=0
ds=pd.read_csv("general_data.csv")
ds.dropna()
ds.drop_duplicates()
le=LabelEncoder()
ds["Attrition"]=le.fit_transform(ds["Attrition"])

# =============================================================================
# #=============================================================================
# matrix=ds[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#         'Education', 'EducationField', 'EmployeeID', 'Gender',
#         'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#         'NumCompaniesWorked', 'Over18', 'PercentSalaryHike',
#         'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#         'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].corr()
# print(matrix)
# #=============================================================================
# stats,p=pear(ds.Attrition,ds.YearsAtCompany)  
# print("the value pf p is: ",p)
# =============================================================================
# =============================================================================
# =============================================================================
# stats,p=pear(ds.Attrition,ds.MonthlyIncome)  
# print("the value pf p is: ",p)
# =============================================================================

# =============================================================================
# stats,p=pear(ds.Attrition,ds.TrainingTimesLastYear)  
# print("the value pf p is: ",p)
# =============================================================================

stats,p=pear(ds.Attrition,ds.YearsWithCurrManager)  
print("the value pf p is: ",p)