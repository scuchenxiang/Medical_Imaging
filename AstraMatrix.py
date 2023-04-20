import astra
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
ImgSize=256
vol_geom = astra.create_vol_geom(ImgSize, ImgSize)
proj_geom = astra.create_proj_geom('parallel', 1, 512, np.linspace(0,2*np.pi,128,False))

# For CPU-based algorithms, a "projector" object specifies the projection
# model used. In this case, we use the "line" model.
proj_id = astra.create_projector('line', proj_geom, vol_geom)#line

# Generate the projection matrix for this projection model.
# This creates a matrix W where entry w_{i,j} corresponds to the
# contribution of volume element j to detector element i.
matrix_id = astra.projector.matrix(proj_id)

# Get the projection matrix as a Scipy sparse matrix.
W = astra.matrix.get(matrix_id)
import scipy.io as sio
X=sio.loadmat("ruijin2my.mat")["X"][:,:,4]
Y=W.dot(X.ravel())
Y=np.reshape(Y,[128,512])
plt.imshow(Y,cmap='gray')
plt.axis("off")
plt.savefig(str(4)+".png",bbox_inches="tight",pad_inches=0.0)
# plt.show()
exit()
sio.savemat("a.mat",{"Y":Y})
# Manually use this projection matrix to do a projection:

P = scipy.io.loadmat('phantom.mat')['phantom256']
# P = scipy.io.loadmat('ruijin2my.mat')['X'][:,:,0]
s = W.dot(P.ravel())
s = np.reshape(s, (len(proj_geom['ProjectionAngles']),proj_geom['DetectorCount']))

plt.gray()
plt.figure("dsfa")
plt.axis('off')
plt.imshow(Y)
plt.show()
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(s)
pylab.show()

# Each row of the projection matrix corresponds to a detector element.
# Detector t for angle p is for row 1 + t + p*proj_geom.DetectorCount.
# Each column corresponds to a volume pixel.
# Pixel (x,y) corresponds to column 1 + x + y*vol_geom.GridColCount.


astra.projector.delete(proj_id)
astra.matrix.delete(matrix_id)