import numpy as np
import scipy.stats as stats
from math import ceil as ceil
from .utils import *

def compute_block_threshold(blocks,p, factor=4,scale=1.):
    '''
    For each block computes x0 s.t. P(|Y|_b >= x0) = 1-p

    @param blocks : np.array([imin,imax,size]) in R^(Nb x 3) : domain block decomposition
    @param p      : probability
    @param factor : int: degree of freedom for each frequency in the chi2 law (default=4)
                            |N_A[i]|^2+|N_E[i]|^2 ~ Chi^2_factor(scale)
    @param scale  : float: scale parameter for the chi2 law  (default=1.)

    @OUTPUT gamma : vector in R^Nb : new block thresholds
    '''
    Nb = blocks.shape[0]
    gamma = 0.*blocks[:,2]
    for k in range(Nb):
        blockSize = blocks[k,2]

        gamma[k] = np.sqrt(blockSize*stats.chi2.isf(1.-p,factor*blockSize, scale=scale))
    return(gamma)

def mergeBranch(tree,i,j, x0, mylist):
    '''
    Merge all branches between branches of indices i and j (i < j)

    @param tree  : [imin | imax | size | Chi2 | x0 | isActive | order ]
    @param i     : int: index min
    @param j     : int: index max
    @param x0    : float : threshold for the new block
    @param mylist: list of int : current list of to-be-suppressed rows in tree
    '''
    tree[i,1] = tree[j,1]
    tree[i,2] = tree[i,1]-tree[i,0]+1
    tree[i,3] = np.sum(tree[i:j+1,3])
    tree[i,4] = x0
    tree[i,5] = (tree[i,3] > tree[i,4])
    tree[i,6] = np.sum(tree[i:j+1,6])

    aaa = [k for k in range(i+1,j+1)]
    mylist.extend(aaa)
    return(1)


def makeTree_BottomUp(y, tree, p, minBlockSize, Rcomp=5., factor=4,scale=1.):
    '''
    Create a block tree by bottom-up process by merging blocks together
    with regard to a threshold defined by probability p

    @param y            : measured data (in Fourier Space)
    @param tree         : tree with minimal identical blocks
                            [imin | imax | size | Chi2 | x0 | isActive | order ]
    @param p            : threshold probability
    @param minBlockSize :
    @param Rcomp        : comparability ratio: blocks b1, b2 are mergeable iif
                              |b1|>|b2| => |b1|/|b2| < Rcomp
    @param factor       : int: degree of freedom for each frequency in the chi2 law (default=4)
                            |N_A[i]|^2+|N_E[i]|^2 ~ Chi^2_factor(scale)
    @param scale        : float: scale parameter for the chi2 law  (default=1.)

    @OUTPUT tree        : tree
    '''

    flag = 0
    cpt = 0
    while (not flag):
        old_tree = np.copy(tree)
        lne2delete = []
        if (cpt == 0):

            x0_4 = stats.chi2.isf(1.-p,4*factor*minBlockSize, scale=scale)
            x0_2 = stats.chi2.isf(1.-p,2*factor*minBlockSize, scale=scale)
            b = 0
            while (b<tree.shape[0]-3):
                # try group by 4
                X2 = np.sum(tree[b:b+4,3])
                if (b == tree.shape[0]-4):
                    n1 = np.sum(tree[b:b+4,2])
                    n2 = np.sum(tree[b:b+2,2]) # = 2*n_block
                    n3 = np.sum(tree[b+2:b+4,2])
                    x0_4 = stats.chi2.isf(1.-p,factor*n1, scale=scale)
                    if (X2 < x0_4):
                        mergeBranch(tree,b,b+3, x0_4,lne2delete)
                        b+=4
                    else: # try group by 2
                        X2_1 = np.sum(tree[b:b+2,3])
                        X2_2 = np.sum(tree[b+2:b+4,3])
                        x0_2_2 = stats.chi2.isf(1.-p,factor*n3, scale=scale)
                        mergeCpt = 0
                        if (X2_1 < x0_2):
                            mergeBranch(tree,b,b+1, x0_2,lne2delete)
                            mergeCpt+=1
                        if (X2_2 < x0_2_2):
                            mergeBranch(tree,b+2,b+3, x0_2,lne2delete)
                            mergeCpt+=1
                        b+=4
                else:
                    if (X2 < x0_4):
                        mergeBranch(tree,b,b+3, x0_4,lne2delete)
                        b+=4
                    else: # try group by 2
                        X2_1 = np.sum(tree[b:b+2,3])
                        X2_2 = np.sum(tree[b+2:b+4,3])
                        mergeCpt = 0
                        if (X2_1 < x0_2):
                            mergeBranch(tree,b,b+1, x0_2,lne2delete)
                            mergeCpt+=1
                        if (X2_2 < x0_2):
                            mergeBranch(tree,b+2,b+3, x0_2,lne2delete)
                            mergeCpt+=1
                        b+=4

            # process last blocks of the tree
            if (b<tree.shape[0]-1):
                #there are 1, 2 or 3 blocks left to process
                n = np.sum(tree[b:,2])
                x0 = stats.chi2.isf(1.-p,factor*n, scale=scale)
                X2 = np.sum(tree[b:,3])
                if (X2 > x0):

                    mergeBranch(tree,b,b+n-1, x0,lne2delete)


        else:
            # try merge 2 by 2
            b = 0
            while (b < tree.shape[0]-1):
                S1 = tree[b,2]
                S2 = tree[b+1,2]
                if (max(S1,S2)/min(S1,S2) < Rcomp):
                    X2 = tree[b,3]+tree[b+1,3]
                    N = tree[b,2]+tree[b+1,2]
                    x0 = stats.chi2.isf(1.-p,factor*N, scale=scale)
                    if (X2 < x0):
                        mergeBranch(tree,b,b+1, x0,lne2delete)
                        b+=2
                    else:
                        b+=1
                else:
                    b+=1

        tree = np.delete(tree,lne2delete,0)

        cpt+=1

        if (old_tree.shape[0] == tree.shape[0]):
            if np.all(old_tree[:,0:3] == tree[:,0:3]):
                flag = 1

    tree[:,3] = norm12_blocks(y, tree[:,:3])**2 #il y a eu des erreurs d'arrondis entre temps.
    return(tree)

def makeTree_BottomUp_init(n_block,Ntot,y,p,factor=4, scale=1.):
    '''
    Creates a regular partition of a Ntot component
    vector with n_block-sized blocks

    @param n_block : size of one block
    @param Ntot    : total size of vector
    @param y       : vector of C^{Nfx2}: measured data (in Fourier Space)
    @param p       : float : probability wanted for the threshold
    @param factor  : int: degree of freedom for each frequency in the chi2 law (default=4)
                            |N_A[i]|^2+|N_E[i]|^2 ~ Chi^2_factor(scale)
    @param scale   : float: scale parameter for the chi2 law  (default=1.)

    @OUTPUT : blocks : 3-column matrix: [imin | imax | block size | Chi2 | x0 | isActive | order]
    '''
    N = ceil(Ntot/n_block) # total number of blocks
    tree = np.zeros((N,7),dtype = object)

    tree[:,0] = np.arange(0,Ntot,n_block)
    tree[:,1] = np.arange(n_block-1,N*n_block,n_block)
    tree[-1,1]= min(Ntot-1,N*n_block-1)
    tree[:,2] = n_block*np.ones(N,dtype=int)
    tree[:,2] = tree[:,1] - tree[:,0] + 1
    tree[:,4] = stats.chi2.isf(1.-p,factor*n_block, scale=scale)*np.ones(N)
    X2 = np.abs(y[:,0])**2+np.abs(y[:,1])**2

    nmax = int(Ntot/n_block)
    if (nmax < Ntot/n_block):
        tree[-1,4] = stats.chi2.isf(1.-p,factor*tree[-1,2], scale=scale)

    else :
        tree[:,1] = np.arange(n_block-1,Ntot+1, n_block)

    tree[:,3] = norm12_blocks(y, tree[:,:3])**2
    tree[:,5] = (tree[:,3] > tree[:,4])
    tree[:,6] = np.ones(N,dtype=int)

    return(tree)
