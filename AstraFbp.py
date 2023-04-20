import scipy.io
import astra
import numpy as np
import pylab
import scipy.io as sio
import matplotlib.pyplot as plt
def AstraFbp(matname,imgname,imgsize,mode,det_len,det_count,theta,source_origin=5,origin_det=10):#不加后面为平行束
    vol_geom = astra.create_vol_geom(imgsize, imgsize)
    proj_geom = astra.create_proj_geom(mode, det_len, det_count,theta)#source_origin,origin_det

    # As before, create a sinogram from a phantom

    P6 = sio.loadmat(matname)[imgname]
    P=P6[:,:,0]
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    sinogram_id, sinogram = astra.create_sino(P, proj_id)
    plt.gray()
    plt.figure(1)
    plt.axis('off')
    plt.imshow(P)
    plt.figure(2)
    plt.axis('off')
    plt.imshow(sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    # create configuration
    cfg = astra.astra_dict('FBP_CUDA')#FBP_CUDA,SART_CUDA,CGLS_CUDA,EM_CUDA
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = { 'FilterType': 'Ram-Lak' }

    # possible values for FilterType:
    # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    # blackman-nuttall, flat-top, kaiser, parzen

    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    rec = astra.data2d.get(rec_id)
    plt.figure(3)
    plt.axis('off')
    plt.imshow(rec)
    plt.show()

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    return rec
if __name__=="__main__":
    # a=astra.create_proj_geom()
    AstraFbp("ruijin2my.mat","X",256,"parallel",1,512,np.linspace(0,2*np.pi,64,False),5,5)
    # AstraFbp("ruijin2my.mat","X",256,"fanflat",4.5/512,512,np.linspace(0,2*np.pi,65,False),5,5)