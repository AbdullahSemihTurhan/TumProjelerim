# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:30:10 2021

@author: mcsem
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler=pd.read_csv("sepet.csv",header=None)
"""
Herhangi bir kolon başliği olmadıgı için header=kolon başlığı None yazıyoruz
"""

####################################################################################################

"""
Not:
    Dosya Duzensiz ve Listesiz Olmasindan oturu
    Dosyayi Listelemek gerekir(Kolon Degil satir Olarak listelememiz Gerekiyor,Her satir farkli liste olacaktir)
"""
##########

"""
Dosyanin Kac farkli Liste olusturulcak onu bilmek gerek 

Suanki 7501 adet
"""
#Bos Bir Liste Olusturuyoruz
t=[]
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])
# t listesine liste halinde string tipinde satırları eklletiyoruz 
# i ve j her bir satir ve sutun 

####################################################################################################
"""
Not:
    apriori Algoritmasında sklern gibi gelişmiş kutuphane olmamasından oturu internetten baska
    kullanicilarin tasarladıgı kutuphaneyi kullanıyoruz.
    
    Bunun için aynı klasör içerside bulunmalı ve alttaki gibi 
    
    from dosya adi import kutuphaneadi 
    
    yazarak çağırıyoruz
    
    bunu Githubdan aldım.
"""
#ctrl+ı = help , argumantasyondur

from apyori import apriori

kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)

print(list(kurallar))

"""
Cıkan degerlerde hangileri ile hangileri eşleşiyor o yazmakta ,lift degerleri de verilmekte
"""














"""
Not:
    Bu algoritma sklearn da yoktur

"""