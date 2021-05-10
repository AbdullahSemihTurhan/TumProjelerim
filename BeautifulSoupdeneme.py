# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:49:02 2021

@author: mcsem
"""

"""
Kütüphaneler
"""
from bs4 import BeautifulSoup 
import requests 
#########################################################################
"""
Web Sitesine Baglan
"""
site=requests.get("https://www.halkyatirim.com.tr/hisse-senedi-skor-kart#grafik-tik")

#kac kod dondu 
print(site.status_code) 
print("#########################################################################")
#Sayfa Kaynagina Bakma
print(site.content)
print("#########################################################################")
#########################################################################
"""
Kaynak Kodlarını al ve BeautifulSoup modülüne aktar
"""


soup=BeautifulSoup(site.content,"lxml") 
"""
lxml kutuphanesi ile BeautifulSoup icine site.content parçalıycaz 
"""

#parcaladıktan sonraki halini gorme
print(soup)
print("#########################################################################")

#########################################################################
"""
BeautifulSoup ile HTML Kodlarını Parçala
:
    Parcalamak icin o linkteki sayfada bulunan secicileri kullanmamız gerekiyor
"""
hazir=soup.find("table",attrs={"class":"table table-hover"})
print(hazir)
#class ı table table-hover" olan butun table ı getir
#########################################################################
"""
Parcalanandan istediklerimizi ayirma
"""

for hisse in hazir:
    print(hazir.a,"\n")














#########################################################################