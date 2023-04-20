import numpy as np
import pandas as pd
from  scipy.fftpack import fft,ifft
from numpy.fft import fft2,ifft2
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import scipy.fft
import matplotlib.image as image

def fun(a,b,c):
    return a[b,c]
if __name__== "__main__" :
    # theta=pd.read_excel( "angles_180.xlsx" )
    # data=pd.read_excel( "projection.xls" ,sheet_name= "s3" ,header =None)
    # Y=data
    A=sio.loadmat("systemMatrix_64_512.mat")["systemMatrix"]
    img=np.reshape(sio.loadmat("ruijin2my.mat")["X"][:,:,0],[-1,1])
    Y=A.dot(img)
    # Y=np.reshape(Y,[512,64])
    Y=np.reshape(Y,[64,512])
    Y=Y.T
    row,col=np.shape(Y)
    # projFft=np.zeros((col,row))
    # for i in range(col):
    #     projFft[i,:]=fft(data[i,:])
    projFft=fft(Y,512,axis=0)
    projFft1=fft(Y,512,axis=0)
    tmp1=list(2*np.linspace(0,row/2-1,256,endpoint=True)/row)
    tmp2=list(2*np.linspace(row,1,256,endpoint=True)/row)
    myfitter=tmp1+tmp2
    projFitter=projFft
    rea=projFft.real
    ima=projFft.imag
    for i in range(64):
        # projFitter[:,i]=rea[:,i]*np.sqrt(np.square(rea[:,i])+np.square(ima[:,i]))
        projFitter[:,i]=projFft[:,i]*myfitter
    # projFitter=np.reshape(projFitter,[180,512]).T
    # projiFft=projFitter
    # for i in range(180):
    #     projiFft[:,i]=ifft(projFitter[:,i])
    projiFft=ifft(projFitter,axis=0)#这一行有问题
    plt.imshow(projiFft.real,cmap= "gray" )
    plt.axis( "off" )
    plt.show()
    M=512
    fbp=np.zeros((M,M))
    realprojiFft=projiFft.real

    #下面这一块区域是正确的，主要是ifft那里的结果不对
    theta = np.linspace(0., 2*180.,64 , endpoint=False)
    for i in range(64):
        th=(theta[i])
        print(i)
        for p in range(M):
            for q in range(M):
                tmp=(p+1-M/2) * np.cos(th) - (q+1-M/2) * np.sin(th)
                t = np.round(tmp)
                r=int(t)
                if r>-512/2 and r<512/2:
                    tttt=round(r+512/2)-1
                    ttt=realprojiFft[round(r+512/2)-1][i]
                    fbp[p][q]+=realprojiFft[round(r+512/2)-1][i]
    fbp = (fbp)/180
    plt.imshow(fbp)
    plt.axis("off")
    plt.show()
    print(fbp)