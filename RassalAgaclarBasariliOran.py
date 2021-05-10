# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:46:25 2021

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

"""
Rassal Agac::
    
    Random Forest:
        aslında birden fazla karar agacının aynı problem icin aynı veri kümesi
        uzerinde cizilmesi ve daha sonra problem çözümünde kullanılmasına dayanıyor
        Çoğunluğun Oyu İsmi veriliyor
        
   
    
    Mantigi:
        Veri Kumesini Birden Fazla Parçaya Bölüp , 
        Her parçadan farklı bir karar agaclari olusturmak
        ardından karar ağaçlarının sonuclarını birleştirmek
        
        
    NOT:
        Verilerin Artmasıyla Basarinin Dusmesi Vardır:
            Cunku Karar Agacindaki Dallanma
            Overfitinge (ezberlemeye ) gidicek Yani HEr bir verinin Karşılık Verisini 
            Ezberleyecek Bu Bir Tehlikedir
        
         
        Bir Diger Durum İse:
            Agacın Çok Fazla Dalın Budaklanması Bu da Daha Buyuk Agaclar ve bu buyuk agaclar uzerinde hesaplama
            zamanı uzaması Gibi Sonuclar Ortaya Cikar
          
   
        RASSAL AĞAÇLAR BU KONULARDA BİRAZ DAHA SIĞI;
        DAHA FAZLA AĞAÇ ÜRETEREK  ÇÖZÜM ÜRETMEYE CALISIYORLAR 
   
    
   
    
"""
#########################################################################################

"""

n_estimators=10 parametresi Kac Tane Karar Agaci Cizilecegini gireriz belirtiriz

"""

from sklearn.ensemble import RandomForestRegressor

ranforreg=RandomForestRegressor(n_estimators=10,random_state=0)

ranforreg.fit(x.values,y.values.ravel())
# x bilgisinden y bilgisini öğren manasına gelir

print(ranforreg.predict([[10]]))

#########################################################################################
"""
Tahmini Gorsellestirme
"""
plt.scatter(x.values,y.values,color="g")

plt.plot(x.values,ranforreg.predict(x.values),color="r")



#########################################################################################

"""
Rassal Agac ile Karar Ağaci Farki :
    
    Karrar agaci önceden verdiğimiz bir cevabdan birini vermek zorunda idi ancak rassal agac sonradan
    kendi türettigi veriyi verir
    
    Rassal Agac birden fazla Karar Agaci İle Olusur
"""












