# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pandas ile Veri Cekmek
veriler = pd.read_csv('veriler.csv') #csv = virgulle ayrilmis veriler
print(veriler)

#veri on isleme
boy=veriler[["boy"]] #boy u list e ekliyor ve sadece boy kolonunu almayı sagliyor
print(boy)




boykilo=veriler[["boy","kilo"]] #sadece boy ve kilo kolonunu almayı saglar
print(boykilo)




class insan:
    boy=180
    def kosmak(self,b):
#class icersine tanimlanan method self isminde ozel bir parametre alir
        return b+10

ali = insan()#insan class indan cagiriyor
print(ali.boy)
print(ali.kosmak(90)) #○az once b olarak verdigimiz parametrenin degeri girilmemis

liste=[1,2,3,4] #liste olusturmak


#Eksik Verileri İsleme Sokmak

eksikveriler=pd.read_csv("eksikveriler.csv")
print(eksikveriler)

#eksik verilerin oldugu yere o kolonun ortalamasını koydurma
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy ='mean')
#'mean' = ortalamasini al
#missing_values = nelere uygulanacak
#strategy = uygulanacak yerlere ne yazayım

yas=eksikveriler.iloc[:,1:4].values
print(yas)

#fit fonksiyonu ogrenilecek olan deger fit fonksiyonu egitmek icin kullanilir 
imputer=imputer.fit(yas[:,1:4]) 
#ustteki ogrenme komutuna gore
#yas in 1 den 4 e kadar olan kolonlarini ogren diyoruz
yas[:,1:4]=imputer.transform(yas[:,1:4])
    #non degerler ogrenilen degerlere degisecek
#FİT İLE OGRENİP , TRANSFORMLA UYGULAMASINI SÖYLÜYORUZ
print(yas)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    