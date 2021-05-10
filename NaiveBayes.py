# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:15:32 2021

@author: mcsem
"""


"""
1.kutuphaneler
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4] #bagımsız degisken

y=veriler.iloc[:,4:5] #bagımlı degisken


X=x.values
Y=y.values


########################################################################
"""
VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

########################################################################
"""
VERİLERİN OLCEKLENMESİ
------------------------
"""
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)

X_test=sc.transform(x_test)

########################################################################

"""
Naive Bayes 
"""

from sklearn.naive_bayes import GaussianNB
"""
Naive Bayes in türleri bulunmaktadır
GaussianNB bu türlerden sadece biridir
GaussianNB dahil 4 türden olusur
"""

gnb=GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
########################################################################
"""
Matrix
"""
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print("Naive Bayes GaussianNB")
print(matrix)
#hata sayı ve oranını gormek icin,matrix olusturmamız gerekiyor , confusion matrix i kullanicaz
#8 sınıftan sadece 1 ini dogru olarak buldu



























































