import numpy as np
from .utils import *

def compute_weights_l1(x_k, gamma_0 ,coeff=1.):
    gamma_A = (gamma_0**2)/(coeff*np.abs(x_k[:,0]) + gamma_0)
    gamma_E = (gamma_0**2)/(coeff*np.abs(x_k[:,1]) + gamma_0)
    return(gamma_A, gamma_E)

def compute_weights(x_k, gamma_0, coeff=1.):
    '''
    Computes new weights for reweighted L1 minimization

    @param x_k     : complex array: normalized result of last resolution in Fourier domain
    @param gamma_0 : initial weights
    @param coeff   : positive scalar - gives more weight to |x_k| in the computation of new coefficients
    '''

    return((gamma_0**2)/(coeff*norm12(x_k) + gamma_0))


#######################################################
###  Block reweighting
#######################################################

def reweighting_blocks(x_k, gamma_b0, blocks, coeff=1.):
    '''
    For block sparsity with L1-reweighting when applying a unique threshold to the whole block
    Computes the new block threshold

    @param x_k      : vector in C^{Nfx2} :current solution in frequency domain
    @param gamma_b0 : vector in R^Nb : block thresholds
    @param blocks   : np.array([imin,imax,size]) in R^(Nb x 3) : freq domain block decomposition
    @param coeff    : positive scalar: reweighting parameter (default=1.)

    @OUTPUT gamma_b : vector in R^Nb : new block thresholds for solution x_k
    '''
    Nb = blocks.shape[0]
    gamma_b = 0.*gamma_b0
    for i in range(Nb):
        imin = blocks[i,0]
        imax = blocks[i,1]
        size = blocks[i,2]
        # norm_block = np.sqrt(np.sum(np.abs(x_k[imin:imax+1,0])**2+np.abs(x_k[imin:imax+1,1])**2) / size)
        norm_block = np.sqrt(np.sum(np.abs(x_k[imin:imax+1,0])**2+np.abs(x_k[imin:imax+1,1])**2))
        gamma_b[i] = gamma_b0[i]**2/(coeff*norm_block+gamma_b0[i])

    return(gamma_b)

def reweighting_blocks_element(x_k, gamma, W_0, blocks, coeff=1.):
    '''
    For block sparsity with L1-reweighting when applying a threshold value per value in each block
    Computes the new weights

    @param x_k    : vector in C^{Nfx2} : current solution in frequency domain
    @param gamma  : vector in R^Nb : block thresholds
    @param W_0    : vector in R^{Nfx2} : initial applied weights
    @param blocks : np.array([imin,imax,size]) in R^(Nb x 3) : freq domain block decomposition
    @param coeff  : positive scalar: reweighting parameter (default=1.)

    @OUTPUT W_n   : vector in R^Nf : new weights to apply for next iteration
    '''

    X = np.sqrt(np.abs(x_k[:,0])**2+np.abs(x_k[:,1])**2)
#     index = (X-gamma > 0.)
#     W_n = np.ones(W_0.shape)
#     W_n[index] = (gamma*W_0[index])**2/(gamma*(coeff*X[index]+gamma*W_0[index]))
    W_n = (gamma*W_0)**2/(gamma*(coeff*X+gamma*W_0))
    return(W_n)

def reweighting_blocks_loc(gamma_b0, blocks, W, x_k):
    '''
    Re-compute block thresholds after application of weight W
    (in order to keep an active block active)

    @param gamma_b0 : vector in R^Nb : initial block thresholds
    @param blocks   : np.array([imin,imax,size]) in R^(Nb x 3) : block decomposition
    @param W        : vector in R^Nf : new weight for each freq value
    @param x_k      : vector in C^{Nfx2} : current solution in frequency domain

    @OUTPUT gamma_b : vector in R^Nb : new block thresholds
    '''
    Nb = blocks.shape[0]
    gamma_b = 0.*gamma_b0
    for i in range(Nb):
        imin = blocks[i,0]
        imax = blocks[i,1]
        size = blocks[i,2]
#         weight_norm = np.sqrt(np.sum(W[imin:imax+1]**2)/size)
        valMax = np.amax(np.abs(x_k[imin:imax+1,:]))
        if (valMax > 0.):
            weight_norm = np.sqrt(np.sum(1./W[imin:imax+1]**2))
            gamma_b[i] = gamma_b0[i]/weight_norm #we want it to be smaller
#             print('BLOCK %d   R = %lf'%(i,1./weight_norm))
        else:
            gamma_b[i] = gamma_b0[i]

    return(gamma_b)
