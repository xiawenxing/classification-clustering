import numpy as np
from scipy.spatial.distance import cdist
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize

def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    '''
    
    N, P = X.shape
    idx = np.zeros((N, 1))
    # Your Code Here
    # Answer begin
    # 1. 初始化第一轮中心点
    centerpoints = np.zeros((k,P))    
    for i in range(k):
        mini = np.min(X[i,:])
        rangei = float(np.max(X[i,:])-mini)
        centerpoints[i,:] = np.mat(mini+rangei*np.random.rand(1,P)) 
    #print(centerpoints)
    # 2. 循环：计算距离，选取新的中心点
    ischanged = True
    time=0
    while ischanged and time<10000:
        time+=1
        ischanged = False
        newclusters = np.zeros([k,P])
        clunum = np.zeros(k)
        for i in range(N):#对每个数据点，都更新其分类
            dist = np.zeros(k)
            for j in range(k):
                dist[j] = np.sqrt(np.sum(np.power(centerpoints[j,:]-X[i,:],2)))
            index = int(np.argwhere(dist==min(dist))[0])
            newclusters[index,:] = np.add(newclusters[index,:],X[i,:])
            clunum[index]+=1
            if idx[i,0] != index :
                idx[i,0] = index
                ischanged = True   
        for i in range(k):
            #centerpoints[i,:] = np.divide(newclusters[i,:] ,clunum[i])             
            ptsInClust = X[np.nonzero(idx[:,0]==i)[0]] 
            centerpoints[i,:] = np.mean(ptsInClust, axis=0)          
    print(idx)        
    # Answer end
    return idx

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    idx = np.zeros((N, 1))
    # Your code here
    # Answer begin
    # 1. 计算D L矩阵    
    pred = np.mat(np.sum(W,axis=0))
    D = np.mat(np.diag((pred.getA())[0]))
    L = D-W
    #Dn = np.power(np.linalg.matrix_power(D,-1),0.5)
    matr = np.dot((D.I),L)
    #matr = np.dot(np.dot(Dn,L),Dn)

    # 2. 求k个最小特征向量组成Y
    lam,H = np.linalg.eig(matr)
    Y = normalize(np.real(H[:,np.argsort(lam)[1:k+2]]))
    # 3. kmeans聚类
    X = Y   
    # Answer end
    X = X.astype(float)
    idx = kmeans(X, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''

    
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')

    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  
        W[i, index] = 1
    W[aj >= threshold] = 0
    W[aj == 0] = 0
    for i in range(N):
        W[i,i] = 0
    return W

'''
dataSet = np.mat([[0,0.8,0.6,0,0.2,0],[0.8,0,0.8,0,0,0],[0.6,0.8,0,0.3,0,0],[0,0,0.3,0,0.8,0.7],[0.2,0,0,0.8,0,0.8],[0,0,0,0.7,0.8,0]])

kmeans(dataSet,3)
print("------------------")
spectral(dataSet,3)
'''


