import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
def ART( A, b, X0, e0=1e-8):
    """
    art algorithm
    :param A: should be a csr matrix ,the speed in csr matrix will be faster than csc matrix
    :param b: m plus 1
    :param X0: n plus 1
    :param e0: mse loss
    :return:
    """
    e=e0+1
    i=0
    while(e>e0):
        tmp=(A[i]@X0-b[i])/(np.dot(A[i],A[i]))*A[i].T
        X=X0-np.reshape(tmp,(-1,1))
        e=np.linalg.norm(A.dot(X)-b)
        X0=X
        i=(i+1)%np.shape(A)[0]
        print(e)
    print(e)
    return X0
def SparseArt(A, b, X0, e0=30):
    """
    art algorithm in sparse form
    :param A: should be a csr matrix ,the speed in csr matrix will be faster than csc matrix
    :param b: m plus 1
    :param X0: n plus 1
    :param e0: mse loss
    :return:
    """
    e=e0+1
    i=0
    row,col=np.shape(A)
    while e>e0  :
        t=(A[i].dot(X0)-b[i])
        X0=X0-(float((A[i].dot(X0)-b[i]))/((A[i].multiply(A[i])).sum())*(A[i].T))
        e=np.linalg.norm(A.dot(X0)-b)
        i=(i+1)%row
        print(e)
    print(e)
    return X0
if __name__=="__main__":
    #稀疏版本的测试
    A=csr_matrix(sio.loadmat("systemMatrix_64_512.mat")["systemMatrix"])#必须是行优先的矩阵
    img=np.reshape(sio.loadmat("phantom.mat")["phantom256"],[-1,1])
    Y=A.dot(img)
    X0=np.zeros((65536,1))
    x=SparseArt(A,Y,X0)
    x=np.reshape(x,[256,256])
    plt.gray()
    plt.figure("img")
    plt.axis('off')
    plt.imshow(x)
    plt.show()
    #非稀疏版本的测试
    # A=np.array([[3,1,9,2],[1,5,7,6],[3,4,9,5],[5,6,4,7]])
    # Y=np.array([[2],[3],[5],[7]])
    # e0=0.000001
    # X0=np.array([[1],[1],[1],[1]])
    # X=ART(A,Y,X0,e0)
    # print((X))
