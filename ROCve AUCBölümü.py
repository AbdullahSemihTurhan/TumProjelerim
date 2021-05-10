# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:23:14 2021

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
########################################################################
"""
KARMASIKLIK MATRİSİ
--------------------
"""

from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)

"""
Parantez icine ne ile ne arasında matrix olusturulacagı yazılıyor

Gerçekte ne oldukları,Tahminde Oldukları
 Aynı verinin gerçeği ve tahmini 
 
 
 Not:
     Koseli Parantez icinde Sayilar Cikicak (tahmin ve test oldugu icin sadece
                                             2 adet koseli parantez cikar)
     
     1. Koseli Parantez Dogru Olanlar
     2. ve diger koseli parantezler yanlisi verir
     
     Ne yapar:
         Gerçek ve Tahmin Degerlerini Basıorum 
         Toplamlarını Basıyorum
         Kac Tane OLdugunu Basıyorum
         Dogru Sayılarını Basıyorum Dİyor
         
     
"""

























