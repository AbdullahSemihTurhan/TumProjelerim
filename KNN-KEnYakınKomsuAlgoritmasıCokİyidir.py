# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:35:39 2021

@author: mcsem
"""

"""
1.kutuphaneler
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4] #bagımsız degisken

y=veriler.iloc[:,4:5] #bagımlı degisken


X=x.values
Y=y.values


########################################################################
"""
VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ

"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

########################################################################
"""
VERİLERİN OLCEKLENMESİ
------------------------
"""
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)

X_test=sc.transform(x_test)

########################################################################
########################################################################

"""
Lojistik Regresyon
---------------
"""

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)

logr.fit(X_train,y_train)
####
"""
Lojistik regresyon Kullanarak Tahmin Yaptirma
"""

y_pred=logr.predict(X_test)

print(y_pred)
print(y_test)
########################################################################
"""
KNN=En Yakın Komsu algoritması
"""

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski' )
"""
Parantez icerisinde yazan 

(kac Komsuya Bakılacak,mesafesi algoritması onemli(minkowski (kendi icerisinde bir algoritmadı
                                                  ss lere bakarak baska algoritmalar kullanılavilir)))


      Karmasıklık Matrixi ve Lojistik Regresyonla Birlikte Kullanılır
"""

knn.fit(X_train,y_train)
#knn icersine fit et ( ogret)
y_pred=knn.predict(X_test)

matrix=confusion_matrix(y_test,y_pred)
#Yeni Matrixe Eitliyor
print(matrix)



















