# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:53:57 2021

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
#*******************************KNN en iyi deger alma***********************************************************
sonuclar=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    # HEr seferinde aynı random state ile baslaması için değer verilmeli
    kmeans.fit(X)
    print(i,"\n",kmeans.cluster_centers_)
    sonuclar.append(kmeans.inertia_)
    #inertia = WCC değerleri
    
    
    
    
 #******************************************************************************************

print("--------------------------K Means grafik (iste baglı)----------------------------------------")

plt.plot(range(1,11),sonuclar)

plt.show()

kmeans=KMeans(n_clusters=4,init="k-means++",random_state=0)
    # HEr seferinde aynı random state ile baslaması için değer verilmeli
y_tahmin=kmeans.fit_predict(X)

print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c="r")
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c="b")
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c="g")
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c="y")

plt.title( "K Means Grafiği")
plt.show()



 #******************************************************************************************

print("--------------------------------------------------------")
"""
Hiyerarşik Bölütleme
"""

from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
y_tahmin=ac.fit_predict(X)
#ac yi uyarlarken ahmin et 
print(y_tahmin)


#3 clusters oldugu icin 3 adet tanımlancak
plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c="r")
#ilk clusters ı gösterdi , (kırmızı)
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c="b")
#ikinci clusters ı gösterdi (mavi)
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c="g")
#ucuncu clusters ı gosterdi
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c="y")

#son clustersı gosterdi (yesil)
plt.title( "Hiyerarşik Bölütleme Grafiği")

plt.show()




 #******************************************************************************************
"""
DENDROGRAM:
    
    sklearn kutuphanesi degil scipy kutuphanesi kullanilir
    
    verinin birbirine baglanma noktalarını gösterir
"""


import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))

plt.show()





