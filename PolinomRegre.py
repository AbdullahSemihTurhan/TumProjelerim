# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:32:41 2021

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
"""
X ile Y yi Gorsellestir ==>
"""

plt.scatter(x.values,y.values,color="red")

"""
X e Karsiik gelen yeni y yi yani tahhmini gorsellestirsin ==>
"""
plt.plot(x,linreg.predict(x.values),color="blue")
plt.show()

#########################################################################################
"""
POLİNOMAL REGRESYON

sklearn Kutuphanesi icinde 

preprocessing Dosyasi icinde 

PolynomialFeatures 

#########################################################################################

Ne İse Yarar : 
    
           Herhangi Bir Sayıyı Polinomal Olarak ifade etmeye yariyor
           
           Istedigimiz Polinom Derecesini Verebiliyoruz
           
           
           
           
           
           
           
Ornek:
    
    polyreg=PolynomialFeatures(degree=2)

    polyreg(objesi) = 2. dereceden bir polinom objesi oldu
    
    
    
degree => derece


en yakın ve uygun dereceyi deneye deneye buluyoruz

"""

from sklearn.preprocessing import PolynomialFeatures

polyreg=PolynomialFeatures(degree=2)


#########################################################################################

"""
Linear Dunyadaki x degerimin Karsiligini
Polinomal Dunyada Cevirirken Yapilanlar
"""

x_polynomaldunya=polyreg.fit_transform(x.values)

print(x_polynomaldunya)

"""
Yazdirdigimizda 

datamizda :
 10 tane satir var o yuzden,1 ve 10 un arasındaki tum sayıların kuvvetleri seklinde yazdirir

3 kolon cikiyor

x in 0. kuvveti 
x in 1. kuvveti
x inn 2. kuuvveti yaziyor  

x= 1 ve 10 arasidaki butun sayilar

degree yi 2 olarak tanimladigimiz icin 2. kuvvetine kadar aliyor

"""
#########################################################################################

#yeninden Linear Regresyon Olusturuyoruz

linreg2=LinearRegression()

linreg2.fit(x_polynomaldunya,y)

"""
linear Regresyonu 

Olusturdugum Polinominal Degiskenleri Kullanarak "y" yi ogren diyoruz

linreg2=> Ogrenme Kutuphanesi
"""

#########################################################################################

plt.scatter(x.values,y.values,color="g")

plt.plot(x.values,linreg2.predict(polyreg.fit_transform(x.values)),color="r")
plt.show()

"""
linreg2 ye polinom seklinde fit etmiştik ondan dolayi,plt olarak cizdirirken:
    
    
    plt.plot(x.values,linreg2.predict(polyreg.fit_transform(x.values)))


   seklinde cizdiriyoruz
"""

#############################################################################################################

"""
TAHMİNLER
"""
print(linreg.predict([[11]]))
"""
Egitim Derecesi 11 iken bu adama ortalama 34716 TL veririm diyor
"""

print(linreg.predict([[5.8]]))
"""
Egitim Derecesi 5.8 iken bu adama ortalama 13688 TL veririm diyor
"""
#############################################################################################################

"""
Daha Onceden Ogrenmedigi Bir Veriyi Tahmin Ettirme:
    
    2. modelde Tahmin Ettiricem
"""



print(linreg2.predict(polyreg.fit_transform([[6.6]])))
"""
Polinomal Derecesine Gore :Egitim Derecesi 6.6 iken bu adama ortalama 10083 TL veririm diyor
"""
print(linreg2.predict(polyreg.fit_transform([[11]])))
"""
Polinomal Derecesine Gore :Egitim Derecesi 11 iken bu adama ortalama 56091 TL veririm diyor
"""








#############################################################################################################


"""
                                         NOTLAR:
"""

#############################################################################################################

"""
Not:
    Herhangi bir Encoder(Kategorik=>Numeric) donusumune ihtiyac duymayacagim icin 
    Data Frame lere,birlestirmelere,Encoderlara ihtiyacım olmiycak dogrudan Egitim
    ve Sistem bolunmesine giricegim
"""

#########################################################################################################

"""
Onemli Not:
       Modeli Daha iyi Kullanmak icin .Butun Veriyi Egitim İcin Kullanacagiz 
       
       Modelin Basarri Durumunu Gozle Gozlemliycem o Yuzden Test sistemi gelistirmiycem
       
       Dolayisiyla Mevcut Verilerin Tamamini Kullanicam o yuzden Verileri Egitim İcin Bolmiycem 
       
       Ondan Dolayi Verilerin Bolunme Islemini Kapattim
       
       Veriyi Olceklemeyecem o yuzden ora da Kapali
    


#########################################################################################################


VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

######################################################################################
VERİLERİN OLCEKLENMESİ
------------------------
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)

y_train=sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
"""

#########################################################################################################

#Sablondan Tek Kullandigim Ozellik Veriyi Okumak
