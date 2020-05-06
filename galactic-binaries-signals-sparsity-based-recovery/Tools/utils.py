import numpy as np
import scipy.stats as stats


def compute_logError(x_th,x_est):
    '''
    Compute NMSE:
    NMSE = -10*log10(||x_th-x_est||/||x_th||)

    @param x_th : theoretic signal
    @param x_est: estimated signal
    '''
    diff = x_th - x_est
    norm2_th  = np.sum(np.abs(x_th)**2)
    norm2_diff = np.sum(np.abs(diff)**2)

    return(-10.*np.log10(norm2_diff/norm2_th))


def find_threshold_chi2(Nfreedom,p,scale=1.):
    '''
    finds x0 so that P(Chi^2_Nfreedom > x0) = 1 - p
    '''
    x0 = stats.chi2.isf(1.-p,Nfreedom, scale=scale)
    return(x0)


def norm12(x):
    '''
    ||x||_{12} = sum_k sqrt{sum_j |x[k,j]|^2}

    @param x : complex array of size Nf*2: x = [Af Ef] with both Af and Ef complex arrays

    @OUT : mixed norm element by element
    '''
    return(np.sqrt(np.abs(x[:,0])**2 + np.abs(x[:,1])**2))


def create_regular_blocks(n_block,Ntot):
    '''
    Creates a regular partition of a Ntot component
    vector with n_block-sized blocks

    @param n_block : size of one block
    @param Ntot    : total size of vector

    @OUTPUT : blocks : 3-column matrix: [imin | imax | block size]
    '''
    K = int(Ntot/n_block)
    if (Ntot > K*n_block):
        K+=1
    blocks = np.zeros((K,3))
    for k in range(K):
        blocks[k,0] = k*n_block  #min index
        blocks[k,1] = min((k+1)*n_block-1,Ntot-1) #max index
        blocks[k,2] = blocks[k,1]-blocks[k,0]+1 #size

    return(np.intc(blocks))

def norm12_blocks(u, blocks):
    '''
    Computes for each block b of blocks:

        ||u||_b12 = sqrt((1/|b|)sum_{i in b} |u[i,0]|^2 + |u[i,1]|^2  )

    @param u      : vector of C^{Nx2}
    @param blocks : block decomposition of C^{Nx2} :
                        3-column matrix [imin | imax | block size]

    @OUTPUT norm  : vector of R+^{Nb} (Nb : number of blocks)
    '''
    K = blocks.shape[0]
    norm = np.zeros(K)
    for i in range(K):
        imin = blocks[i,0]
        imax = blocks[i,1]
        size = blocks[i,2]
        # norm[i] = np.sqrt(np.sum(np.abs(u[imin:imax+1,0])**2+np.abs(u[imin:imax+1,1])**2)/size)
        norm[i] = np.sqrt(np.sum(np.abs(u[imin:imax+1,0])**2+np.abs(u[imin:imax+1,1])**2))

    return(norm)

def stdDev_estimator(y, N=50):
    '''
    Computes an estimate of signal y standard deviation using Median Absolute
    Deviation estimator (MAD) over sliding window of size 2*N:

    sigma[k] = median(|X-median(X)|)/0.67449
    where X = y[k-N:k+N]

    @param y : real signal
    @param N : half size of window (in number of used points)

    @OUTPUT sigma : empiric standard deviation for signal y
    '''

    sigma = 0.*y
    Ntot = sigma.shape[0]

    for k in range(N,Ntot-N):
        X = y[k-N:k+N]
        medX = np.median(X)
        sigma[k] = np.median(np.abs(X-medX))/0.67449

    # sigma[:N]=sigma[N]*np.ones(N)
    # sigma[-N:]=sigma[-N]*np.ones(N)

    return(sigma)
