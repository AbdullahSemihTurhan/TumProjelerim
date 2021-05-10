# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:17:15 2021

@author: mcsem
"""

"""
1.kutuphaneler
"""
#8102 satir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
2.VERİ ON İSLEME
"""
#2.1 VERİ YUKLEME

veriler=pd.read_excel('USD_TRY-Gemi-Verileri.xlsx') 
print(veriler)


"""
Egitim Seviyesini X 
Maaslari Y olarak Tanimlayacagim
===>
"""

x=veriler[["Açılış"]]
print(x)

y=veriler[["Fark "]]
print(y)

X=x.values

#########################################################################################
from sklearn.ensemble import RandomForestRegressor

ranforreg=RandomForestRegressor(n_estimators=10,random_state=0)

ranforreg.fit(x.values,y.values.ravel())
# x bilgisinden y bilgisini öğren manasına gelir

print(ranforreg.predict([[6]]))

#########################################################################################
"""
Tahmini Gorsellestirme
"""
plt.scatter(x.values,y.values,color="g")

plt.plot(x.values,ranforreg.predict(x.values),color="r")


