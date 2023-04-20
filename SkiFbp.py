import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon,iradon_sart,iradon
import scipy.io as sio
def skiFbp(imgname,Y,theta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    sinogram = radon(imgname, theta=theta, circle=True)

    print(np.shape(sinogram))
    Y=np.reshape(Y,[64,-1])
    Y=Y.T
    reconstruction_fbp = iradon((Y), theta=theta, circle=True,output_size=256)
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title( "Radon transform\n(Sinogram)" )
    ax2.set_xlabel( "Projection angle (deg)" )
    ax2.set_ylabel( "Projection position (pixels)" )
    ax2.imshow(Y, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    fig.tight_layout()
    plt.show()
if __name__=="__main__":
    X=sio.loadmat("ruijin2my.mat")["X"][:,:,0]
    # system=sio.loadmat("parsysMat.mat")["sysmat"]
    # Y=system.dot(np.reshape(X,[-1,1]))
    # Y=np.reshape(Y,[-1,64])
    Y=sio.loadmat("a.mat")["Y"]
    plt.gray()
    plt.figure("dsfa")
    plt.axis('off')
    plt.imshow(X)
    plt.show()
    skiFbp(X,Y,theta = np.linspace(0., 2*180.,64 , endpoint=False))