# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:56:52 2021

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
#########################################################################################

"""
Makine Ogrenmesi Baslatma
"""
from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(x.values,y.values)
"""
<== x e bakarak Y yi ogren

.values sebebi = Verileri Data Frame Verdigimiz İcin Onun Degerlerini Cevirir:
     Pandas Olarak verilen Verileri Numpy Dizisi Haline çevirdik
"""
#########################################################################################
from sklearn.metrics import r2_score
print("Linear regresyon Degerlendirme")
print(r2_score(y.values, linreg.predict(x.values)))


"""
 bu Sekilde Yazilir y = oğrenmeye calisilan , ogrendigini uygulamaya calisan
 
 1 e ne kada yakın olursa o kadar iyi
"""


