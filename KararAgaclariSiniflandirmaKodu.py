# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
############################################################
"""
KUTUPHANELER
"""
############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################
"""
Veri Onisleme
"""
############################################################
veriler=pd.read_csv("veriler.csv")
x=veriler.iloc[:,1:3]
y=veriler.iloc[:,3:4]

X=x.values
Y=y.values

############################################################
"""
Model Olusturma
"""
############################################################
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
############################################################
"""
Verilerin Ayrıstırılması
"""
############################################################
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
############################################################
"""
Karar Agaclari Siniflandirma
"""
############################################################
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy")
"""
Parantez icersine aldigi Parametreyi sklearn Sitesinde Dökümantasyonundan Kopyala 
Yapiştir Yapıyoruz
"""
dtc.fit(X_train,y_train)
#xtrain den y train i öğren
y_pred=dtc.predict(X_test)

#X testdeki verileri al , sonra tahmin yap
##############################################################
"""
Matrix 

sonucu karsilastirmak icin kullaniyoruz
"""
#############################################################"""

from sklearn.metrics import confusion_matrix

matrix=confusion_matrix(y_test,y_pred)
#y test ile Y pred i karsilastir
print("Karar Agaci Siniflandirma")
print(matrix)




























