import numpy as n
import pandas as pd
import matplotlib.pyplot as m
count=0
var=pd.read_csv("general_data.csv")
var.dropna()
var.drop_duplicates()
#number of males
# =============================================================================
# for i in var["Gender"]:
#    # c=var.at[i,'Gender']
#     if(i=="Male"):
#         count +=1
# print("male count is count is: ",count)
# =============================================================================
#number of females
# =============================================================================
# for i in var["Gender"]:
#    # c=var.at[i,'Gender']
#     if(i=="Female"):
#         count +=1
# print("Female count is count is: ",count)
# =============================================================================
# =============================================================================
# mean=var[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#        'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
#        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
#        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].mean()
# print(mean)
# =============================================================================
# =============================================================================
# mode=var[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#        'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
#        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
#        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].mode()
# print(mode)
# =============================================================================
# =============================================================================
# for i in range(0,4410):
#     g=var.at[i,'Gender']
#     a=var.at[i,'Attrition']
#     if(g=="Female" and a=="Yes"):
#         count +=1
# print(count)
# =============================================================================
# =============================================================================
# for i in range(0,4410):
#     g=var.at[i,'BusinessTravel']
#     a=var.at[i,'Attrition']
#     if(g=="Travel_Rarely" and a=="Yes"):
#         count +=1
# print(count)
# for column in var.columns:
#     if var[column].dtype == object:
#         print(str(column) + ' : ' + str(var[column].unique()))
#         print(var[column].value_counts())
#         print("_________________________________________________________________")
#    
# =============================================================================
# =============================================================================
# box_plot=var.MonthlyIncome
#     
# m.boxplot(box_plot)
# m.hist(
# m.show()
# =============================================================================
# =============================================================================
# des=var[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#        'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
#        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
#        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].describe()
# print(des)
# varience=var[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#        'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
#        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
#        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].var()
# print(varience)
# =============================================================================
# =============================================================================
# skew=var[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
#        'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
#        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
#        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
#        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
#        'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].skew()
# print(skew)
# =============================================================================
# =============================================================================
# =============================================================================
# plot1=var.Age
# m.boxplot(plot1)
# =============================================================================
# =============================================================================
# plot2=var.MonthlyIncome
# m.boxplot(plot2)
# =============================================================================
# =============================================================================
# plot3=var.DistanceFromHome
# m.boxplot(plot3)
# =============================================================================
plot4=var.MonthlyIncome
m.boxplot(plot4)