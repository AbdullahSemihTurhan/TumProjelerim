# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:39:18 2021

@author: mcsem
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

"""
Veri On isleme
"""
#sci-kit learn
"""
Kategorik(string) => Numeric(int) dönusumu
"""
#ulkeler Numeric yapıldı
ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe=preprocessing.OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()

print(ulke)
#cinsiyet numeric yaoıldı
c=veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])

print(c)

ohe=preprocessing.OneHotEncoder()

c=ohe.fit_transform(c).toarray()

print(c)
"""
Numpy Dizileri Data Frame Donusumu
"""

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])
print(sonuc3)

"""
Data Frame Birleştirme
"""
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)
"""
verilerin egitim ve test icin bolunmesi
"""
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)
"""
#
verilerin olceklenmesi
--------------------------------------
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
"""




from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train) # xtrain bağımsız değişkeninden ytrain i öğren

y_pred=regressor.predict(x_test)# x test ayrılmış kısmı üstteki algoritmaya göre predict et ve ypred e yaz


"""
<= Cinsiyet Buldurmak 
Boy Kolonunu Buldurmak=>
"""
boy=s2.iloc[:3:4].values #♦3 ren sonraki 4 ü al 
print(boy)

sol=s2.iloc[:,:3] #soldan 3 e kadar al
sag=s2.iloc[:,4:] # sağdan 4 e kadar al


veri=pd.concat([sol,sag],axis = 1) # pd.concaat=birlestirmek


x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0) #verileri kullanarak veri yi bul 


r2=LinearRegression()

r2.fit(x_train,y_train) # xtrain bağımsız değişkeninden ytrain i öğren

y_pred=r2.predict(x_test)# x test ayrılmış kısmı üstteki algoritmaya göre predict et ve ypred e yaz


"""
Not:
    Kod y_test deki verileri y_pred de tahmin ediyor , y_pred deki veriler y_test e ne kadar yakınsa sistem o kadar iyidir
    
    predict = tahmin etmek anlamına gelir 
    
    
    
    LinearRegression.fit(x_train,y_train) # xtrain bağımsız değişkeninden ytrain i öğren
    y_pred = LinearRegression.predict(x_test)
    
    ya da 
    
    r2=LinearRegression()
    r2.fit(x_train,y_train) # xtrain bağımsız değişkeninden ytrain i öğren
    y_pred=r2.predict(x_test)
    
    seklinde tahmin ettirilebilir
"""






"""

MODELİN BASARI KRİTERLERİ BELİRLENMESİ VE P-VALUE HESAPLAMA 

"""

"""
Modelin Başarısıyla ilgili sistem kurmak
"""

import stadtsmodels.api as sm

#degiskenlerin hangisinin sisteme daha fazla etkiledigini görebilmek icin bu degiskenleri iceren bir dizi uzerinden gidicem
# amacım bir dizi olusturucam dizinin icine butun degiskenleri koycam sonra sirasiyla degiskenleri eleyerek gidicem
#hangi degiskenin p value si yuksekse ve hangi degisken sistemi bozuyorsa eliycez


X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
 #22 satır 1 kolonn a 1 ekleyecek , sabit degisken icin , axis 1 dedik cubku
#kolon olarak eklemesini istiyoruz,

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
#p si buyuk olan deger burdan elenir

X_l=np.array(X_l,dtype=float)

model=sm.OLS(boy,X_l).fit() #boy(aranan) ile dizi uzerindeki baglantıyı kurucaz

print(model.summary())

# belge gibi p degeri raporu cıkarır,buyuk olan elenir




















