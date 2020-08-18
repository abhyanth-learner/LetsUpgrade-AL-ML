import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
random_state = np.random.RandomState(42)




# IMPORT PYODD PACKAGES AND THE METHOODS
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging


#IMPORTING METRICS PACKAGES

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from time import time





#DEFINE DATA FILE AND READ X AND Y

mat_file_list=["arrhythmia.mat","cardio.mat","glass.mat","ionosphere.mat","letter.mat","lympho.mat","mnist.mat","musk.mat","optdigits.mat","pendigits.mat","pima.mat","satellite.mat","satimage-2.mat","shuttle.mat","vertebral.mat","vowels.mat","wbc.mat"]



#DEFINE NINE OUTLIER DETECTION TOOLS TO BE COMPARED`
df_columns=['Data','#Samples','# Dimensions','Outlier Perc','ABOD','CBLOF','FB','HBOS','IForest','KNN','LOF','MCD','OCSVM','PCA']



#HOW TO READ .MAT FILE
data=loadmat("cardio.mat")
#x is independant and y is dependant

#ROC PERFORMANCE EVOLUTION TABLE roc_auc_score
roc_df=pd.DataFrame(columns=df_columns)

# PRECITION SCORE  precision_n_scores
prn_df=pd.DataFrame(columns=df_columns)
#TIME DATAFRAME
time_df=pd.DataFrame(columns=df_columns)



#EXPLORING ALL MAT FILES
# for  mat_file in mat_file_list:
        
#     print(mat_file)
#     mat=loadmat(mat_file)
#     X= mat["X"]
#     y=mat["y"].ravel()#converts 2d to 1d
#     outliers_fraction=np.count_nonzero(y)/len(y)
#     outlier_percentage=round(outliers_fraction*100,ndigits=4)
    
    
#     #constructing containers for saving the result
#     roc_list=[mat_file[:-4],X.shape[0],X.shape[1],outlier_percentage]
#     prn_list=[mat_file[:-4],X.shape[0],X.shape[1],outlier_percentage]
#     time_list=[mat_file[:-4],X.shape[0],X.shape[1],outlier_percentage]
#     Xtrain,Xtest,Ytrain, Ytest=train_test_split(X,y,test_size=0.3,random_state=random_state)
#     X_train_norm,y_test_norm=standardizer(Xtrain,Xtest)
#     classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
#        contamination=outliers_fraction),
#        'Cluster-based Local Outlier Factor': CBLOF(
#            contamination=outliers_fraction, check_estimator=False,
#            random_state=random_state),
#        'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
#                                          random_state=random_state),
#        'Histogram-base Outlier Detection (HBOS)': HBOS(
#            contamination=outliers_fraction),
#        'Isolation Forest': IForest(contamination=outliers_fraction,
#                                    random_state=random_state),
#        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
#        'Local Outlier Factor (LOF)': LOF(
#            contamination=outliers_fraction),
#        'Minimum Covariance Determinant (MCD)': MCD(
#            contamination=outliers_fraction, random_state=random_state),
#        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
#        'Principal Component Analysis (PCA)': PCA(
#            contamination=outliers_fraction, random_state=random_state),
#    }

#   for clf_name, clf in classifiers.items():
#     t0 = time()
#     clf.fit(X_train_norm)
#     test_scores = clf.decision_function(X_test_norm)
#     t1 = time()
#     duration = round(t1 - t0, ndigits=4)
#     time_list.append(duration)

#     roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
#     prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
#     print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
#              'execution time: {duration}s'.format(
#            clf_name=clf_name, roc=roc, prn=prn, duration=duration))

#     roc_list.append(roc)
#     prn_list.append(prn)

# temp_df = pd.DataFrame(time_list).transpose()
# temp_df.columns = df_columns
# time_df = pd.concat([time_df, temp_df], axis=0)

# temp_df = pd.DataFrame(roc_list).transpose()
# temp_df.columns = df_columns
# roc_df = pd.concat([roc_df, temp_df], axis=0)

# temp_df = pd.DataFrame(prn_list).transpose()
# temp_df.columns = df_columns
# prn_df = pd.concat([prn_df, temp_df], axis=0)
from time import time
random_state = np.random.RandomState(42)

for mat_file in mat_file_list:
   print("\n... Processing", mat_file, '...')
   mat = loadmat( mat_file)

   X = mat['X']
   y = mat['y'].ravel()
   outliers_fraction = np.count_nonzero(y) / len(y)
   outliers_percentage = round(outliers_fraction * 100, ndigits=4)

   # construct containers for saving results
   roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
   prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
   time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

   # 60% data for training and 40% for testing
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                       random_state=random_state)

   # standardizing data for processing
   X_train_norm, X_test_norm = standardizer(X_train, X_test)

   classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
       contamination=outliers_fraction),
       'Cluster-based Local Outlier Factor': CBLOF(
           contamination=outliers_fraction, check_estimator=False,
           random_state=random_state),
       'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
                                         random_state=random_state),
       'Histogram-base Outlier Detection (HBOS)': HBOS(
           contamination=outliers_fraction),
       'Isolation Forest': IForest(contamination=outliers_fraction,
                                   random_state=random_state),
       'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
       'Local Outlier Factor (LOF)': LOF(
           contamination=outliers_fraction),
       'Minimum Covariance Determinant (MCD)': MCD(
           contamination=outliers_fraction, random_state=random_state),
       'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
       'Principal Component Analysis (PCA)': PCA(
           contamination=outliers_fraction, random_state=random_state),
   }

   for clf_name, clf in classifiers.items():
       t0 = time()
       clf.fit(X_train_norm)
       test_scores = clf.decision_function(X_test_norm)
       t1 = time()
       duration = round(t1 - t0, ndigits=4)
       time_list.append(duration)

       roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
       prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

       print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
             'execution time: {duration}s'.format(
           clf_name=clf_name, roc=roc, prn=prn, duration=duration))

       roc_list.append(roc)
       prn_list.append(prn)

   temp_df = pd.DataFrame(time_list).transpose()
   temp_df.columns = df_columns
   time_df = pd.concat([time_df, temp_df], axis=0)

   temp_df = pd.DataFrame(roc_list).transpose()
   temp_df.columns = df_columns
   roc_df = pd.concat([roc_df, temp_df], axis=0)

   temp_df = pd.DataFrame(prn_list).transpose()
   temp_df.columns = df_columns
   prn_df = pd.concat([prn_df, temp_df], axis=0)



























