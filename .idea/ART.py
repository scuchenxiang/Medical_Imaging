import numpy as np
def ART( A, b, X0, e0=1e-6):
    e=e0+1
    i=0
    while(e>e0):
        tmp=((A[i]@X0-b[i])/(np.linalg.norm(A[i]))*(A[i].T/np.linalg.norm(A[i])))
        X=X0-np.reshape(tmp,(-1,1))
        e=np.linalg.norm(X-X0)
        X0=X
        i=(i+1)%np.shape(A)[0]
    print(e)
    return X0
