# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

veriler=pd.read_csv('eksikveriler.csv')
print(veriler)

#2.2 VERİ ON İSLEME

boy=veriler[["boy"]]
print(boy)

boykilo=veriler[["boy","kilo"]]
print(boykilo)

x=10
"""
class isim:
    boy=180
    def fonksiyonisim (self,b):
        return b+10
sinifaozneeklemek=isim()
istedigimizkadareklenebilir=isim()
print(sinifaozneeklemek.boy)
print(sinifaozneeklemek.kosmak(90))
"""
#2.3 EKSİK VERİLER

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy ='mean')

yas=eksikveriler.iloc[:,1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4]) 
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas) 
""""
Ustteki İmputer ile eksik yerleri doldurur
"""
ulke = veriler.iloc[:,0:1].values
print(ulke)
"""
Ustteki 
encoder:Kategorik(Nominal - Ordinal) Degerleri => i Numeric Degerlere Cevirir
"""

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
#Birbirinden Farkli Ulkelere Sirasiyla 0 , 1 ,2 veriyor
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)


"""
Kolon basliklari,etikketleri tasimak ve her etiketin altina 1 veya sifir dierek
oraya ait veya degil demektir

One.HotEncoder()
"""
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
#0,1,2 olarak siralanmıs ulke kodlarini 000 olarak veriyor 
#bbahsettigi ulke kolonunda 1 vererek yaziyor
#OneHotEncoder bu ise yarar


"""
Numpy Dizileri Data Frame Dönüşümü
"""
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)
"""
DATAFRAME / DİZİLERİN FARKI
DATAFRAME dizilerden en buyuk farki index ve kolon basliklarinin olmasidir

"""

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
"""
cinsiyeti de veri kümesinden veriler.iloc ile sokuyoruz
"""


sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)


"""
pd.concat= birleştirmek 
"""
#DataFrame Birleştirme Etiketi

s=pd.concat([sonuc,sonuc2],axis=1)

"""
axis = 0 
kolon basliklarindan tutusan yerleri esitler
axis = 1
satir basliklarindan tutusan yerleri esitler 
"""
print(s)

s1=pd.concat([s,sonuc3],axis=1)
print(s1)
"""
s=ulke boy kilo ve yas olan kolonlar 
sonuc3=cinsiyet

"s" i kullanarak sonuc 3 u bulmaya calisir , cinsiyeti bulmaya calisir
"""
"""
DETAYLI ANLATIMI JUPYTER NOTEBOOK TA YAZMISDIM
"""


########################################################################
"""
VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

"""
VERİLERİN OLCEKLENMESİ
------------------------
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)

y_train=sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
"""
"""
model inşaası 
"""
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)



















