# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:14:01 2021

@author: mcsem
"""


"""
1.kutuphaneler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
2.VERİ ON İSLEME
"""
#2.1 VERİ YUKLEME

veriler=pd.read_csv('maaslar.csv')
print(veriler)


"""
Egitim Seviyesini X 
Maaslari Y olarak Tanimlayacagim
===>
"""

x=veriler.iloc[:,1:2]
print(x)

y=veriler.iloc[:,2:3]
print(y)

"""
DataFrame Olarak Gelen Veriyi Numpy Array a hazir cevirme
"""

X=x.values
Y=y.values

#########################################################################################

"""
Makine Ogrenmesi Baslatma
"""
from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(x.values,y.values)

#########################################################################################


plt.scatter(x.values,y.values,color="red")

plt.plot(x,linreg.predict(x.values),color="blue")
plt.show()


#############################################################################################################
"""
Verilerin Olceklenmesi
"""

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_olcekli=sc.fit_transform(X)

"""
Y icin Ayrı bir  Scaler olusturma
"""
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

#############################################################################################################
"""
Destek Vektör Regresyon Kodlaması

Destek Vektör Regresyonu = SVR 
ingilizcesinin kisaltilmisi
"""

from sklearn.svm import SVR

svrreg=SVR(kernel='rbf')

svrreg.fit(x_olcekli,y_olcekli)

"""
Gosterme
"""
plt.scatter(x_olcekli,y_olcekli,color="r")

plt.plot(x_olcekli,svrreg.predict(x_olcekli),color="b")
"""
<==
 x olcekli deger icin x olcekli degerin tahmin karsiligini bul 
 ve göster

"""
#############################################################################################################
"""
SVR olarak Tahmin Ettirme
"""

print(svrreg.predict(11))
print(svrreg.predict(6.6))








