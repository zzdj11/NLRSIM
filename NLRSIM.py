import numpy as np
from lshash.lshash import LSHash
from scipy.io import loadmat
from correlation_pearson.code import CorrelationPearson
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.io import savemat
data = loadmat('E:/桌面/scData/Data_Bre.mat')
in_X = data['in_X']
labels = data['true_labs']
in_X = np.transpose(in_X)

a, b = in_X.shape
flag = (in_X != 0).astype(int).sum(axis=1)
list = []
for i in range(a):
    if flag[i] > (b*0.05):
         list.append(i)
X = in_X[list,:]
m, n = X.shape 

[nclass] = np.max(labels, axis=0)
lsh = LSHash(hash_size=1, num_hashtables=50, input_dim=m)

W = np.zeros((n, n))
corr = CorrelationPearson()
for i in range(n):
    lsh.index(X[:, i], extra_data='%s' % i)


for j in range(n):
    for res in lsh.query(X[:, j]):
        sim_ = int(res[0][-1])
        # cor_ = corr.result(X[:, j], X[:, sim_])
        W[j, sim_] = 1
        W[sim_, j] = 1
