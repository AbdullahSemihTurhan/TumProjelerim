# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:53:16 2021

@author: mcsem
"""


"""
1.kutuphaneler
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler=pd.read_excel('iris.xls')

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
"""
BURDAN İTİBAREN SINIFLANDIRMA ALGORİTMALARI BASLAR
"""
########################################################################
"""
1. Lojistik Regresyon
---------------
"""

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix #KARMASIKLIK MATRİSİ
logr=LogisticRegression(random_state=0)

logr.fit(X_train,y_train)
####
"""
Lojistik regresyon Kullanarak Tahmin Yaptirma
"""

y_pred=logr.predict(X_test)

print(y_pred)
print(y_test)
print("Lojistik Regresyon Sonucu \n")

matrix=confusion_matrix(y_test,y_pred)
print(matrix)
########################################################################
"""
2. KNN Algoritması
"""


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print("KNN Algoritması Sonucu \n")
matrix=confusion_matrix(y_test,y_pred)
print(matrix)

########################################################################
"""
3. SVC
"""

from sklearn.svm import SVC
svc=SVC(kernel='poly')
svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)

print("SVC Algoritması Sonucu \n")
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
########################################################################
"""
4. NaiveBayes
"""

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)

print("Naive Bayes Algoritması Sonucu \n")
matrix=confusion_matrix(y_test,y_pred)
print(matrix)

########################################################################
"""
5. Desicion Tree (Karar Agaci)
"""

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)

print("Desicion Tree (Karar Agaci) Algoritması Sonucu \n")
matrix=confusion_matrix(y_test,y_pred)
print(matrix)

########################################################################
"""
6. Random Forest ( Rassal Orman )
"""

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy') #10 kademeden olusan bir orman 
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)

#####################
"""
Proba
"""
#########################
y_proba=rfc.predict_proba(X_test)


print("Random Forest ( Rassal Orman ) Algoritması Sonucu \n")
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
print("RandomForest Proba\n",y_proba)
########################################################################

print(y_test)

"""
ROC 
"""

from sklearn import metrics

fpr,tpr,thold=metrics.roc_curve(y_test, y_proba[:,0],pos_label='e')
#e olanları pozitif olarak almak için , 'k' yazılırsa kızları pozitif verir 
"""
Y_proba sadece 1 kolon bastırabilir o yuzden diğer kolonları atoyurz
"""
"""
Ustteki Şekilde yazılır
"""
print("False Positive")
print(fpr)
print("True positive")
print(tpr)
########################################################################
"""
fpr:
    False Positive ler
    
tpr : 
    True Positive ler
    
    
"""
########################################################################
"""
Proba Nedir:
    
    Bastırdığı değerler Olasılık Değerleridir
    
    Degerin True olma ve False Olma ihtimalleri
    
    Örnek:
        
        [0.9 0.1]
 [0.3 0.7]
 [0.4 0.6]
 [0.3 0.7]
 [0.5 0.5]
 [0.3 0.7]
 [0.4 0.6]
 [0.6 0.4]
 
 bu değer 
 1. satır %90 orana karşılık %10 oran vermiş 
 
 
    
    
    
    
    
    
    
    
    
    """
