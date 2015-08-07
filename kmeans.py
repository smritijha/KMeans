'''
author: smriti
'''

import pandas
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = pandas.read_csv('ds_100k.csv', usecols=[8,16,17,18,19,20,21,22,23])
data = data.fillna(0)

data['featured'] = data['featured'].replace('f',0).replace('t',1)

X = preprocessing.normalize(data.values, axis=1, norm='l1') #l1 normalize

est = KMeans(n_clusters = 2,init='k-means++')
labels = est.fit_predict(X)

print "Average total shares for Cluster 1 (Engaging)"
print X[labels==1, 1].mean()
print "Average total shares for Cluster 2 (Non-engaging)"
print X[labels==0, 1].mean()

plt.figure(3)
plt.scatter(X[:,5], X[:,1], c=labels.astype(np.float))
plt.xlabel('FK_Grade')
plt.ylabel('Shares')

plt.figure(2)
plt.scatter(X[:,6], X[:,1], c=labels.astype(np.float))
plt.xlabel('Fog_Index')
plt.ylabel('Shares')

plt.figure(1)
plt.scatter(X[:,4], X[:,8], c=labels.astype(np.float))
plt.xlabel('Words')
plt.ylabel('Likes')

plt.figure(4)
plt.scatter(X[:,4], X[:,2], c=labels.astype(np.float))
plt.xlabel('Words')
plt.ylabel('Comments')

fig = plt.figure(5)
ax = Axes3D(fig, elev=40, azim=210)
ax.scatter(X[:,4], X[:,1], X[:,6], c=labels.astype(np.float))
ax.set_xlabel('Words')
ax.set_ylabel('Number of shares')
ax.set_zlabel('Fog Index')

plt.show()
