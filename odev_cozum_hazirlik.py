#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")

"""

 Not:
    OneHotEncoder = 2 den fazla sık icirn ornegin burda hava durumu gibi 
    ancak dogru yanlis ve yes no da label encoder kullanilir
    
    2 den fazla ise OneHotEncoder
    2 veya 1 ise Label Encoder

#veri on isleme

---------------------------------------------------------------------------


hava=veriler.iloc[:,0:1].values
print(hava)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
hava[:,0]=le.fit_transform(veriler.iloc[:,0])

print(hava)

ohe=preprocessing.OneHotEncoder()

hava=ohe.fit_transform(hava).toarray()


---------------------------------------------------------------------------





ruzgar=veriler.iloc[:,-2:-1].values
print(ruzgar)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ruzgar[:,-2:-1]=le.fit_transform(veriler.iloc[:,-2:-1])

print(ruzgar)













---------------------------------------------------------------------------








oyun=veriler.iloc[:,-1:].values
print(oyun)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
oyun[:,-1]=le.fit_transform(veriler.iloc[:,-1])

print(oyun)



---------------------------------------------------------------------------



Numpy Dizileri Data Frame Donusumu


havadu=pd.DataFrame(data=hava,index=range(22),columns=['sunny','overcast','rainy'])
print(havadu)



ruzga=pd.DataFrame(data=ruzgar,index=range(22),columns=['FALSE','TRUE'])
print(ruzga)


oyu=pd.DataFrame(data=oyun,index=range(22),columns=['yes','no'])
print(ruzga)


---------------------------------------------------------------------------

"""




###############################################################






"""

Ustteki islem yerine Kısaca Alttaki gibi yazılabilir
----------------------------------------------
encoder:  Kategorik -> Numeric
---------------------------------------------------------------------------
"""


from sklearn import preprocessing
veriler2= veriler.apply(preprocessing.LabelEncoder().fit_transform)

#Butun kolonlara Label Encoder olarak veriler 2 nin ustune apply et                        
#veriler 2 acıldıgında kolayca otomatik apply ediyor
#Ancak sadece en bastaki hava durumunu OneHotEncoding olarak kaydetmemiz lazım
#o yuzden onu ayircaz simdi

c = veriler2.iloc[:,:1]
#SADECE 1 İNCİ KOLONU SECTİK
#MANUEL OLARAK ONEHOTENCODER A CEVİRDİK 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)
"""
---------------------------------------------------------------------------
verileri toplamak=>
"""
havadurumu=pd.DataFrame(data=c,index=range(14),columns=['o','r','s'],axis=1)
#bas harflerini yazdim yeterli olacaktir
#♦14 data oldugu icin 14 dedik
#axis = 1 cunku kolon bazinda istiyoruz
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
#usttekinde 1 ve 2 alinacagi icin 1 den 3 e kadar yaziyoruz
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)
#♥usttekinde son verilerden 2 kolonu aliyoruz 
"""
---------------------------------------------------------------------------
EGİTİM VE BOLME
"""
#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

#SONVERİLERDEKİ BENİM BAGIMSIZ DEGISKENLERIM SONVERILER DIZISININ BASLANGICTAN SONA KADAR OLANLAR
#SON KOLON HARIC

"""
BAGIMLI DEGISKEN = Y = BULUNMASI ISTENEN
"""
"""
---------------------------------------------------------------------------
SISTEM BASLIYCAK TAHMIN EDECEK
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







