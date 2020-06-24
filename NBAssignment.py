# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:34:50 2020

@author: RADHIKA
"""
####################Salary data Assignment####################

###Using salaryData_Train
import pandas as pd
import numpy as np
salarydata_train=pd.read_csv("D:\\ExcelR Data\\Assignments\\Naive Bayes\\SalaryData_Train.csv")
salarydata_test=pd.read_csv("D:\\ExcelR Data\\Assignments\\Naive Bayes\\SalaryData_Test.csv")

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
salarydata_train.columns
salarydata_test.columns
salarydata_train.shape
salarydata_test.shape
salarydata_train.isnull().sum
salarydata_test.isnull().sum
salarydata_train.head
salarydata_test.head
salary_columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in salary_columns:
    salarydata_train[i] = number.fit_transform(salarydata_train[i])
    salarydata_test[i] = number.fit_transform(salarydata_test[i])
colnames = salarydata_train.columns
len(colnames[0:13])
trainX=salarydata_train[colnames[0:13]]
trainY=salarydata_train[colnames[13]]
testX=salarydata_test[colnames[0:13]]
testY=salarydata_test[colnames[13]]

ignb = GaussianNB() # normal distribution
pred_gnb = ignb.fit(trainX,trainY).predict(testX)

confusion_matrix(testY,pred_gnb)
#([[10759,   601],
 #      [ 2491,  1209]], dtype=int64)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209))#80%

mnnb = MultinomialNB()
pred_mnb=mnnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,pred_mnb)
#array([[10891,   469],
 #      [ 2920,   780]], dtype=int64)

print ("Accuracy",(10891+780)/(10891+469+2920+780))

##Accuracy 0.7749667994687915

##GaussianNB is the best model accuracy is 80%

########SMS DATA ASSIGNMENT################

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB

sms=pd.read_csv("C:\\Users\\RADHIKA\Desktop\\sms_raw_NB.csv",encoding = "ISO-8859-1")
sms.columns
sms.shape
sms.head
sms.isnull().sum
sms.describe()

messages=["Hi there", "Hi, how are you","Hello"] 
count_vect = CountVectorizer()
sparse_matrix = count_vect.fit_transform(messages)
count_vect.get_feature_names()
#['are', 'hello', 'hi', 'how', 'there', 'you']
sparse_matrix.toarray()
#array([[0, 0, 1, 0, 1, 0],
#       [1, 0, 1, 1, 0, 1],
#       [0, 1, 0, 0, 0, 0]], dtype=int64)
#Like that i tranform my dataset column text
sparse_matrix = count_vect.fit_transform(sms["text"])
len(count_vect.get_feature_names())
#8698
sparse_matrix.toarray().shape
# (5559, 8698)
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix , sms['type'], test_size=0.2) 
model = MultinomialNB().fit(X_train, y_train)
y_predict = model.predict(X_test)
print (confusion_matrix(y_test, y_predict))
#[[959  17]
# [  6 130]]
print (accuracy_score(y_test,y_predict))

###0.97931654676259
pd.crosstab(y_test, y_predict,rownames=['Actual'], colnames=['Predicted'],  margins=True)
#Predicted  ham  spam   All
#Actual                    
#ham        959    17   976
#spam         6   130   136
#All        965   147  1112











