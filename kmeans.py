# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:34:38 2021

@author: mcsem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler=pd.read_csv("musteriler.csv")



X=veriler.iloc[:,3:].values #Maas ve hacim aldik


"""

K Means Modul Cagirma
"""
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init="k-means++")

"""
3 cluster lı 

k-means ++ yöntemini kullanarak
"""

kmeans.fit(X)
print(kmeans.cluster_centers_)

"""
cluster larını nerde olusturdugunu yaz

yazılan ilk değer hacim diğer değer maas
"""
print("--------------------------------------------------------------")
#######################################
"""
Fazladan olarak K için en iyi degeri nasıl bulabiliriz d
böyle ======>
"""
sonuclar=[]
for i in range (1,10):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    # HEr seferinde aynı random state ile baslaması için değer verilmeli
    kmeans.fit(X)
    print(i,"\n",kmeans.cluster_centers_)
    sonuclar.append(kmeans.inertia_)
    #inertia = WCC değerleri
    print("\n",sonuclar)
    
    
print("------------------------------------------------------------------")


"""
Sonucu Gorsellestirelim
"""

plt.plot(range(1,10),sonuclar)














