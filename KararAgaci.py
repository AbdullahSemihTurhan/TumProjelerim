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
Karar AgaciRegresyonu

Random Prametresi alir
"""

from sklearn.tree import DecisionTreeRegressor

DeTreeRe=DecisionTreeRegressor(random_state=0)

DeTreeRe.fit(x.values,y.values)

plt.scatter(x.values,y.values,color="b")

plt.plot(x.values,DeTreeRe.predict(x.values),color="r")

"""
<=

Makine Ogrenmesinde Yaptigimiz Egitimi Karar Agaci Regresyonu uzerinde uyguladi
ve Sonucu Grafik olarak gösterttik sonuc çok başarılı oldu.

Karar agaci Regresyonunda Objeleri Scaler Etmeye GGerek yoktur
"""

#########################################################################################
"""
Karar Agaci Regresyonu istedigim numarayi Tahmin Ettirme
"""

print(DeTreeRe.predict([[11]]))
print(DeTreeRe.predict([[6.6]]))





#########################################################################################
"""
Karar Ağacı Nasıl işliyor:
    
    Bir ağac dali gibi 2 ye bölüne bölüne gidiyor bu sayede daha yakın rakamlar buluyor
    EkranGörüntüsü olarak kaydettim inceele
    --------------------------------------------------------
Karar Agaci :
    
    girilen her ne olursa olsun sonuc eski sonuclardan biri olucak,hic
    olmayan bie sonuc vermeyecektir
"""


















