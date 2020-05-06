import numpy as np
from .utils import *


def prox_f(u, gamma,alpha=1.):
    '''
    Returns the solution of problem  :
                argmin_x alpha*gamma*||x||_12 + (1/2)*||u - x||_2^2

        where x = [A E] in C^(nx2)
    Finds joint solution for [A E] vector

    @param u       : complex array of initial signal (noisy) in frequency domain
    @param gamma_n : real array of threshold values (frequency domain)
    '''
    u1 = u[:,0]
    u2 = u[:,1]
    X = np.sqrt(np.abs(u1)**2 + np.abs(u2)**2) - alpha*gamma
    coeff = alpha*gamma/X + 1.

    index = (X > 0.)
    realPart_1 = np.zeros(X.shape) #0.*X
    imagPart_1 = np.zeros(X.shape) #0.*X
    realPart_1[index] = np.real(u1[index])/coeff[index]
    imagPart_1[index] = np.imag(u1[index])/coeff[index]
    realPart_2 = np.zeros(X.shape) #0.*X
    imagPart_2 = np.zeros(X.shape) #0.*X
    realPart_2[index] = np.real(u2[index])/coeff[index]
    imagPart_2[index] = np.imag(u2[index])/coeff[index]
    s1 = realPart_1 + imagPart_1*1.j
    s2 = realPart_2 + imagPart_2*1.j
    N = len(s1)

    return(np.concatenate((s1.reshape(N,1),s2.reshape(N,1)), axis=1))




#######################################################
### FUNCTIONS FOR L1 MINIMIZATION - Block Sparsity
#######################################################
def prox_f_block(u, blocks, gamma_b, freq, alpha=1., W=None, Nb_max=10):
    '''
    Analytic computation of the proximal function for a domain decomposed in blocks


    @param u       : input vector of C^{Nx2}
    @param blocks  : np.array([imin,imax,size]) in R^(Nb x 3) : domain blocks decomposition
    @param gamma_b : block thresholds: in R^Nb (threshold per block)
    @param freq    : frequency vector
    @param alpha   : dilatation coeff
    @param W       : weights to apply in frequency domain. (Default W = None)
    @param Nb      : maximal size of an active block (Default: Nb = 10)

    @OUTPUT z      : z = prox_{alpha*f}(u)
    '''
    if (W is not None):
        # reweighting
        U = 0.*u
        U[:,0] = u[:,0]*W
        U[:,1] = u[:,1]*W
    else:
        U = np.copy(u)

    Nb = blocks.shape[0]
    z = 0.*U
    for i in range(Nb):
        imin = blocks[i,0]
        imax = blocks[i,1]
        size = blocks[i,2]
        # Xb = np.sqrt(np.sum(np.abs(U[imin:imax+1,0])**2+np.abs(U[imin:imax+1,1])**2) / size) - alpha*gamma_b[i]/size
        Xb = np.sqrt(np.sum(np.abs(U[imin:imax+1,0])**2+np.abs(U[imin:imax+1,1])**2)) - alpha*gamma_b[i]
        if ((Xb > 0.) and (size <= Nb_max)):
            # z[imin:imax+1,:] = U[imin:imax+1,:] / (1.+alpha*gamma_b[i]/(size*Xb))
            z[imin:imax+1,:] = U[imin:imax+1,:] / (1.+alpha*gamma_b[i]/Xb)

#     index = (freq>=1.e-2)
#     z[index,:] = 0.*z[index,:]

    if (W is not None):
        z[:,0] = z[:,0]/W
        z[:,1] = z[:,1]/W

    return(z)
