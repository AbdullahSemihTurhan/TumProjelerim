# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:48:22 2021

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


from sklearn.ensemble import RandomForestRegressor

ranforreg=RandomForestRegressor(n_estimators=10,random_state=0)

ranforreg.fit(x.values,y.values.ravel())
# x bilgisinden y bilgisini öğren manasına gelir
a=int(input("Tahmin Edilcek Degeri Giriniz..."))
print(ranforreg.predict([[a]]))

#########################################################################################
"""
Tahmini Gorsellestirme
"""
plt.scatter(x.values,y.values,color="g")

plt.plot(x.values,ranforreg.predict(x.values),color="r")



#########################################################################################

