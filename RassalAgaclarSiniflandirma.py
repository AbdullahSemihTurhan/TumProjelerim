# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:58:21 2021

@author: mcsem
"""

############################################################
"""
KUTUPHANELER
"""
############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################
"""
Veri Onisleme
"""
############################################################
veriler=pd.read_csv("veriler.csv")
x=veriler.iloc[:,1:3]
y=veriler.iloc[:,3:4]

X=x.values
Y=y.values

############################################################
"""
Model Olusturma
"""
############################################################
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
############################################################
"""
Verilerin Ayrıstırılması
"""
############################################################
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
############################################################
"""
Rassal Agaclar Siniflandirma
"""
############################################################
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
"""
10 dal olacagını o kadar ogrenmesi gerektigini belirttik
Parantez icersine aldigi Parametreyi sklearn Sitesinde Dökümantasyonundan Kopyala 
Yapiştir Yapıyoruz
"""
rfc.fit(X_train,y_train)
#xtrain den y train i öğren
y_pred=rfc.predict(X_test)

#X testdeki verileri al , sonra tahmin yap
##############################################################
"""
Matrix 

sonucu karsilastirmak icin kullaniyoruz
"""
#############################################################"""

from sklearn.metrics import confusion_matrix

matrix=confusion_matrix(y_test,y_pred)
#y test ile Y pred i karsilastir
print("Rassal Agac Siniflandirma")
print(matrix)
