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

print(svrreg.predict(1))
print(svrreg.predict(6))

#############################################################################################################
"""
R2.Score:
    
    Regressyonun Dogruluk Derecesini Bulmamiza yarar
    eğer derece 1 e ne kadar yakınsa o kadar iyidir 1 ise kusursuzdur
    
    0 altında ise asırı berbatdır
    
    0 ile 1 arasında olur deger arttıkca o kadar iyidir
       
"""

from sklearn.metrics import r2_score

print("\nDestek VektorRegresyonu R2 Degeri\nSVR Sistemin Egitilmis Derecesi:\n")
print(r2_score(y_olcekli,svrreg.predict(x_olcekli)))

"""
r2_score(r2_score(y_olcekli,svrreg.predict(x._olcekli))

r2_score(Ogretilmek istenen datalar , Öğrettiğimiz regresyonun tahmin etme kalıbı(regresyon farketmez bu ornektir))
#########################################################################################

sistem 1 eğitilmiş çok iyi (1 ile 0 arasında olur 1 e yaklastikca iyilesir)
#########################################################################################
print yaparak direk calistiriyoruz yeterli
#########################################################################################

R2 = Sistemin Ogrenmislik Derecesi 
"""







