import numpy as np
import scipy.io as sio
import torch
import scipy
import os
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def Sart(A, b, X0, e0=1e-10,maxit=10000):
    """
    :param A: maxtrix,m*n
    :param b: m*1
    :param X0: init result
    :param e0: max Residual,mse loss
    :param maxit: max iter number
    :return:
    """
    e=e0+1
    it=0
    row,col=np.shape(A)
    Ai=np.sum(A,axis=0,keepdims=True)
    Aj=np.sum(A,axis=1,keepdims=True)
    while it<maxit and e>e0:
        # tmp=np.dot(A.T,(np.dot(A,X0)-b)/(np.sum(A**2,axis=1).reshape(row,1)))
        # tttt=(np.dot(A,X0)-b)
        X=X0+np.dot(A.T,(b-np.dot(A,X0))/(Aj))/(Ai.T)
        e=np.linalg.norm(A.dot(X)-b)
        X0=X
        it+=1
        print(e)
    return X0
def SparseSart(A, b, X0, e0=1e-8,maxit=1000):
    e=e0+1
    it=0
    row,col=np.shape(A)
    Ai=A.sum(axis=0)
    Aj=A.sum(axis=1)
    while it<maxit and e>e0:
        X0=X0+A.T.dot((b-A.dot(X0))/Aj)/(Ai.T)
        e=np.linalg.norm(A.dot(X0)-b)
        it+=1
        print(e)
    print(e)
    return X0
if __name__=="__main__":
    #sart稀疏版本的测试
    A=sio.loadmat("systemMatrix_64_512.mat")["systemMatrix"]
    img=np.reshape(sio.loadmat("phantom.mat")["phantom256"],[-1,1])
    Y=A.dot(img)
    # init result
    X0=np.zeros((65536,1))

    x=SparseSart(A,Y,X0)
    x=np.reshape(x,[256,256])
    plt.gray()
    plt.figure("sart result")
    plt.axis('off')
    plt.imshow(x)
    plt.show()
    print(x)

    #SART非稀疏版本的测试
    # A=np.array([[3,1,9,2],[1,5,7,6],[3,4,9,4]])
    # Y=np.array([[2],[3],[5]])
    # # e0=0.000001
    # X0=np.array([[1],[1],[1],[1]])
    # print(Sart(A,Y,X0))