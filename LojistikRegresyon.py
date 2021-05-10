# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:58:11 2021

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

"""
<==
x train i öğren ve uygula 
x test için yeniden ogrenme train den ogrendigini uygula 
anlamına gelir
"""
"""
fit_transform() = egittiğini uygula
transform() = Uygulama islemi
"""
########################################################################


"""
Lojistik Regresyon
---------------
"""

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)

logr.fit(X_train,y_train)
####
"""
Lojistik regresyon Kullanarak Tahmin Yaptirma
"""

y_pred=logr.predict(X_test)

print(y_pred)
print(y_test)




























