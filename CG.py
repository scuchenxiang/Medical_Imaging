import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
def CG(A,b,tol=1e-6,maxit=10):
    """
    :param A: matrix m*n
    :param b: m*1
    :param tol: max residual
    :param maxit: max iter number
    :return: x n*1
    """
    col,row=np.shape(A)
    # if col!=row:
    #     b=np.dot(A.T,b)
    #     A=np.dot(A.T,A)
    xk=np.zeros((np.shape(A)[0],1))
    rk=np.dot(A,xk)-b
    pk=-rk
    a=np.dot(rk.T,pk)
    it=0
    while np.linalg.norm(rk)>tol and it<maxit:

        # algfak=-np.dot(rk.T,pk)/(np.dot(pk.T,np.dot(A,pk)))
        # xk=xk+algfak*pk
        # rk1=np.dot(A,xk)-b
        # betak1=np.dot(rk1.T,np.dot(A,pk))/np.dot(pk.T,np.dot(A,pk))
        # pk=-rk1+betak1*pk
        # rk=A@xk-b#残差
        # pk=-rk#共轭方向
        algfak=np.dot(rk.T,rk)/((np.dot(np.dot(pk.T,A),pk)))#最优步长
        xk1=xk+algfak*(pk)
        t1=np.dot(A,pk)
        t2=algfak*(np.dot(A,pk))
        rk1=rk+algfak*(np.dot(A,pk))
        betak1=np.dot(rk1.T,rk1)/(np.dot(rk.T,rk))
        pk1=-rk1+betak1*(pk)
        pk=pk1
        rk=rk1
        xk=xk1
        it+=1
        print(np.linalg.norm(rk1))
    # print(rk)
    return xk
if __name__== "__main__" :
    #稀疏版本的测试
    # A=sio.loadmat("systemMatrix_64_512.mat")["systemMatrix"]
    # img=np.reshape(sio.loadmat("ruijin2my.mat")["X"][:,:,0],[-1,1])
    # Y=A.dot(img)
    # X0=np.zeros((65536,1))
    # x=CG(A,Y,X0)
    # x=np.reshape(x,[256,256])
    # plt.gray()
    # plt.figure("dsfa")
    # plt.axis('off')
    # plt.imshow(x)
    # plt.show()
    # print(x)


    #非稀疏版本的测试
    A=np.array([[3,2,8],[2,4,5],[8,5,7]])
    Y=np.array([[2],[3],[5]])
    # e0=0.000001
    X0=np.array([[0],[0],[0]])
    res=np.dot(np.linalg.inv((A)),(Y))
    print(res)
    print(np.linalg.norm(A.dot(res)-Y))
    x1=CG(A,Y)
    print(x1)
    print(np.linalg.norm(A.dot(x1)-Y))
