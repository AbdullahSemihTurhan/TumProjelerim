# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:19:47 2021

@author: mcsem
"""
#Kutuphaneler
import numpy as np
import padas as pd
import matplotlib.pyplot as plt

# Veri Yukleme
veriler=pd.read_csv("veriler.csv")

x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values

#Verilerin Egitim ve test Bolunmesi
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#Verilerin Olceklenmesi

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)

X_test= sc.transform(x_test)


"""
Tahmin Regresyon Modellerini Secip Burdan itibaren yazılır
"""


