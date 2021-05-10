# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:52:18 2021

@author: mcsem


HAZIR SABLON UZERİNDE DEGİSİKLİK YAPMAYA UYGUN 
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

aylar=veriler[["Aylar"]]
print(aylar)

satislar=veriler[["Satislar"]]
print(satislar)

satislar2 = veriler.iloc[:,:1].values

print(satislar2)
"""
verilerin egitim ve test icin bolunmesi
"""
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
"""
verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
"""





"""
MODEL İNŞASI (LİNEAR REGRESSİON)
"""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin= lr.predict(x_test)

x_train=x_train.sort_index() #INDEX E GORE SİRALA DEMEKTİR
y_train=x_train.sort_index() #INDEX E GORE SİRALA DEMEKTİR
plt.plot(x_train,y_train)
## xtrain deki her bir satıra y train deki baska bbir sattır gelir

"""
predict = tahmin demektir
"""





"""
INDEXE GORE DEGİLDE VERİLERE GORE SIRALANSA İDİ EN KUCUK AY EN KUCUK SATİS İLE EŞLEŞİRDİ
ONU ONMELEK İCİN VERİLERE GORE DEĞİL İNDEX E GORE SİRALANMASİ GEREKİR.
"""


























