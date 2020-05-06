import numpy as np
import matplotlib.pyplot as plt
font = {'family': 'monospace', #'serif','monospace',
        'weight': 'bold', #'normal'
        'size': 14}
plt.rc('font',**font)

import matplotlib as mpl

###############################################
###  FREQUENCY DOMAIN PLOTS                 ###
###############################################

def plot_freq(freq, Adf, Edf, F0=None,xlim=None, ylim=None,
                title=None, ylabel=None, save=0, picDir=None):
    '''
    Log-Log plot (freq, Adf) and (freq, Edf) with matching axis

    @param freq   : array of frequencies
    @param Adf    : array of 1st signal in freq domain
    @param Edf    : array of 2nd signal in freq domain
    @param F0     : real - frequency to highlight in the plot
    @param xlim   : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim   : [ymin, ymax] - sets limits of y axis of the plots
    @param title  : "my title" - plot title
    @param ylabel : ["Adf", "Edf"] - labels for y axis
    @param save   : 0 (don't save) or 1 (save)
    @param picDir : "/path/to/my/picture/directory/"
    '''
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].loglog(freq, np.abs(Adf),linewidth=2)
    ax[0].grid(True)
    ax[1].loglog(freq, np.abs(Edf),linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)

    if (title==None):
        title = "Frequency Domain"

    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$|\hat{A}|$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$|\hat{E}|$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0])
        ax[1].set_ylabel(ylabel[1])

    if (F0 != None):
        ax[0].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
        ax[1].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.17)
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)
    plt.show()
    return(fig,ax)

def plot_freq_XYZ(freq, Xf, Yf, Zf, F0=None,xlim=None, ylim=None,
                title=None, ylabel=None, save=0, picDir=None):
    '''
    Log-Log plot (freq, Xf), (freq, Yf) and (freq, Zf) with matching axis

    @param freq   : array of frequencies
    @param Xf     : array of 1st signal in freq domain
    @param Yf     : array of 2nd signal in freq domain
    @param Zf     : array of 3rd signal in freq domain
    @param F0     : real - frequency to highlight in the plot
    @param xlim   : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim   : [ymin, ymax] - sets limits of y axis of the plots
    @param title  : "my title" - plot title
    @param ylabel : ["Xf", "Yf", "Zf"] - labels for y axis
    @param save   : 0 (don't save) or 1 (save)
    @param picDir : "/path/to/my/picture/directory/"
    '''
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].loglog(freq, np.abs(Xf),linewidth=2)
    ax[0].grid(True)
    ax[1].loglog(freq, np.abs(Yf),linewidth=2)
    ax[1].grid(True)
    ax[1].loglog(freq, np.abs(Zf),linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)

    if (title==None):
        fig.suptitle("Frequency Domain")
    else:
        fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$|\hat{X}|$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$|\hat{Y}|$',fontweight='bold',fontsize=18)
        ax[2].set_ylabel(r'$|\hat{Z}|$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[2],fontweight='bold',fontsize=18)
    if (F0 != None):
        ax[0].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
        ax[1].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
        ax[2].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
        ax[2].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)
        ax[2].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)
    plt.show()
    return(fig,ax)

def plot_freq_filter(freq, Adf, Edf, gamma_n, F0=None, xlim=None, ylim=None,
                title=None, ylabel=None, filterlabel=None, save=0, picDir=None):
    '''
    Log-Log plot (freq, Adf) and (freq, Edf) with matching axis, displaying the threshold defined by gamma_n

    @param freq        : array of frequencies
    @param Adf         : array of 1st signal in freq domain
    @param Edf         : array of 2nd signal in freq domain
    @param gamma_n     : array of threshold values in freq domain
    @param F0          : real - frequency to highlight in the plot
    @param xlim        : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim        : [ymin, ymax] - sets limits of y axis of the plots
    @param title       : "my title" - plot title
    @param ylabel      : ["Adf", "Edf"] - labels for y axis
    @param filterlabel : "gamma_n" - label of filter plot
    @param save        : 0 (don't save) or 1 (save)
    @param picDir      : "/path/to/my/picture/directory/"
    '''
    mpl.rcParams['agg.path.chunksize'] = 10000
    if (filterlabel==None):
        filterlabel = r'$\gamma_n$'

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].loglog(freq, np.abs(Adf),linewidth=2)
    ax[0].loglog(freq, gamma_n, '--g', label = filterlabel,linewidth=2)
    ax[0].grid(True)
    ax[1].loglog(freq, np.abs(Edf))
    ax[1].loglog(freq, gamma_n, '--g', label = filterlabel,linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)
    # plt.subplots_adjust(right=0.8,bottom=0.15)
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)

    ax[1].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    ax[0].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    if (title==None):
        title = "Frequency Domain"

    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$|\hat{A}|$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$|\hat{E}|$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)

    if (F0 != None):
        ax[0].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
        ax[1].loglog([F0, F0],[1.e-20, 1.e-15], '--k',linewidth=2)
    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)


def plot_freq_comparison(freq, Adf0, Edf0, Adf, Edf, xlim=None, ylim=None,
                title=None, ylabel=None, plotLabel=None, save=0, picDir=None):
    '''
    Log-Log comparative plot {(freq, Adf0),(freq, Adf)} and {(freq, Edf0),(freq, Edf)} with matching axis

    @param freq      : array of frequencies
    @param Adf0      : array of 1st signal of reference in freq domain
    @param Edf0      : array of 2nd signal of reference in freq domain
    @param Adf       : array of 1st signal in freq domain
    @param Edf       : array of 2nd signal in freq domain
    @param F0        : real - frequency to highlight in the plot`
    @param xlim      : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim      : [ymin, ymax] - sets limits of y axis of the plots
    @param title     : "my title" - plot title
    @param ylabel    : ["Adf", "Edf"] - labels for y axis
    @param plotLabel : ["reference", "result of the computation"] - labels of
                        the reference and the to-be-compared signals (Adf0/Adf or Edf0/Edf)
    @param save      : 0 (don't save) or 1 (save)
    @param picDir    : "/path/to/my/picture/directory/"
    '''
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))
    if (plotLabel==None):
        plotLabel=['REF','Noisy']

    ax[0].loglog(freq, np.abs(Adf0),'g',label=plotLabel[0],linewidth=2)
    ax[0].loglog(freq, np.abs(Adf),'r',label=plotLabel[1],linewidth=2)
    ax[0].grid(True)

    ax[1].loglog(freq, np.abs(Edf0),'g',label=plotLabel[0],linewidth=2)
    ax[1].loglog(freq, np.abs(Edf),'r',label=plotLabel[1],linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)

    # plt.subplots_adjust(right=0.8,bottom=0.15)
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)

    ax[1].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    ax[0].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))


    if (title==None):
        title = "Frequency Domain"

    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$|\hat{A}|$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$|\hat{E}|$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)

    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)

def plot_freq_modAE(freq, U, gamma, gamma_n=None, xlim=None, ylim=None,
                title=None, ylabel=None, plotLabel=None, save=0, picDir=None):
    '''
    Log-Log plot {(freq, Y)}  displaying thresholds gamma and gamma_n

    @param freq      : array of frequencies
    @param U         : array of signals [Af Ef] in freq domain
    @param xlim      : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim      : [ymin, ymax] - sets limits of y axis of the plots
    @param title     : "my title" - plot title
    @param ylabel    : [r'\sqrt{|\hat{A}|^2 + |\hat{E}|^2}'] - labels for y axis
    @param plotLabel : ["gamma_0", "gamma_n"] - labels of the different thresholds
    @param save      : 0 (don't save) or 1 (save)
    @param picDir    : "/path/to/my/picture/directory/"
    '''
    Y = np.sqrt(np.abs(U[:,0])**2 + np.abs(U[:,1])**2)
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True,figsize=(8.5,5))
    if (plotLabel==None):
        plotLabel=['gamma','gamma_n']

    ax.loglog(freq, Y,linewidth=2)
    ax.loglog(freq, gamma,'--g',label=plotLabel[0],linewidth=2)

    if (gamma_n is not None):
        ax.loglog(freq, gamma_n,'--r',label=plotLabel[1],linewidth=2)

    ax.grid(True)

    ax.set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)
    # plt.subplots_adjust(right=0.8,bottom=0.15)
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)

    ax[1].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    ax[0].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    if (title==None):
        title = "Frequency Domain"

    fig.suptitle(title)

    if (ylabel==None):
        ax.set_ylabel(r'$\sqrt{|\hat{A}|^2 + |\hat{E}|^2}$',fontweight='bold',fontsize=18)
    else:
        ax.set_ylabel(ylabel[0],fontweight='bold',fontsize=18)

    if (xlim != None):
        ax.set_xlim(xlim)
    if (ylim != None):
        ax.set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()

    return(fig,ax)


def plot_freq_blocktree(freq, Adf, Edf, blocks,xlim=None, ylim=None,
                title=None, ylabel=None, save=0, picDir=None):
    '''
    Log-Log plot (freq, Adf) and (freq, Edf) with matching axis

    @param freq   : array of frequencies
    @param Adf    : array of 1st signal in freq domain
    @param Edf    : array of 2nd signal in freq domain
    @param blocks : array of Nbx3 [imin|imax|size] (division of freq space)
    @param F0     : real - frequency to highlight in the plot
    @param xlim   : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim   : [ymin, ymax] - sets limits of y axis of the plots
    @param title  : "my title" - plot title
    @param ylabel : ["Adf", "Edf"] - labels for y axis
    @param save   : 0 (don't save) or 1 (save)
    @param picDir : "/path/to/my/picture/directory/"
    '''
    mpl.rcParams['agg.path.chunksize'] = 10000

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].loglog(freq, np.abs(Adf),linewidth=2)
    ax[0].grid(True)
    ax[1].loglog(freq, np.abs(Edf),linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$f$ (Hz)',fontweight='bold',fontsize=18)

    if (title==None):
        title = "Frequency Domain"

    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$|\hat{A}|$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$|\hat{E}|$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)


    # plot blocks
    Nb = blocks.shape[0]
    ymin = np.amin(np.abs(Adf))
    ymax = np.amax(np.abs(Edf))
    for k in range(Nb):
        imin = blocks[k,0]
        imax = blocks[k,1]
        xb = [freq[imin],freq[imax],freq[imax],freq[imin],freq[imin]]
        yb = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xb,yb, c='coral', alpha=0.5)

    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)

###############################################
###  TIME DOMAIN PLOTS                      ###
###############################################

def plot_time(tm, At, Et,
                xlim=None,ylim=None,
                title=None, ylabel=None, save=0, picDir=None):
    '''
    Plot (tm, At) and (tm, Et) with matching axis

    @param tm     : array of time values
    @param At     : array of 1st signal values in time domain
    @param Et     : array of 2nd signal values in time domain
    @param title  : "my title" - plot title
    @param ylabel : ["At", "Et"] - labels for y axis
    @param save   : 0 (don't save) or 1 (save)
    @param picDir : "/path/to/my/picture/directory/"
    '''
    norm_time = 60.*60.*24. #to get time in days

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].plot(tm/norm_time, At,linewidth=2)
    ax[0].grid(True)
    ax[1].plot(tm/norm_time, Et,linewidth=2)
    ax[1].grid(True)

    ax[1].set_xlabel(r'$t$ (days)',fontweight='bold',fontsize=18)

    if (title==None):
        title = "Time Domain"
    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$A$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$E$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)

    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)

def plot_time_XYZ(tm, Xt, Yt, Zt,
                title=None, ylabel=None, save=0, picDir=None):
    '''
    Plot (tm, Xt), (tm, Yt) and (tm, Zt) with matching axis

    @param tm     : array of time values
    @param Xt     : array of 1st signal values in time domain
    @param Yt     : array of 2nd signal values in time domain
    @param Zt     : array of 3rd signal values in time domain
    @param title  : "my title" - plot title
    @param ylabel : ["Xt", "Yt", "Zt"] - labels for y axis
    @param save   : 0 (don't save) or 1 (save)
    @param picDir : "/path/to/my/picture/directory/"
    '''
    norm_time = 60.*60.*24. #to get time in days

    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True,figsize=(8.5,5))

    ax[0].plot(tm/norm_time, Xt,linewidth=2)
    ax[0].grid(True)
    ax[1].plot(tm/norm_time, Yt,linewidth=2)
    ax[1].grid(True)
    ax[2].plot(tm/norm_time, Zt,linewidth=2)
    ax[2].grid(True)

    ax[1].set_xlabel(r'$t$ (days)',fontweight='bold',fontsize=18)

    if (title==None):
        fig.suptitle("Time Domain")
    else:
        fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel("X",fontweight='bold',fontsize=18)
        ax[1].set_ylabel("Y",fontweight='bold',fontsize=18)
        ax[2].set_ylabel("Z",fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)
        ax[2].set_ylabel(ylabel[2],fontweight='bold',fontsize=18)
    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)

def plot_time_comparison(tm, At0, Et0, At, Et,
                xlim=None,ylim=None, title=None, ylabel=None,
                plotLabel=None,save=0, picDir=None):
    '''
    Comparative plot {(tm, At0),(tm, At)} and {(tm, Et0), (tm, Et)} with matching axis

    @param tm        : array of time values
    @param At0       : array of 1st reference signal values in time domain
    @param Et0       : array of 2nd reference signal values in time domain
    @param At        : array of 1st signal values in time domain
    @param Et        : array of 2nd signal values in time domain
    @param xlim      : [xmin, xmax] - sets limits of x axis of the plots
    @param ylim      : [ymin, ymax] - sets limits of y axis of the plots
    @param title     : "my title" - plot title
    @param ylabel    : ["At", "Et"] - labels for y axis
    @param plotLabel : ["reference", "result of the computation"] - labels of
                        the reference and the to-be-compared signals (At0/At or Et0/Et)
    @param save      : 0 (don't save) or 1 (save)
    @param picDir    : "/path/to/my/picture/directory/"
    '''
    norm_time = 60.*60.*24. #to get time in days

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,figsize=(8.5,5))

    if (plotLabel==None):
        plotLabel=['REF','Noisy']
    ax[0].plot(tm/norm_time, At0, 'g', label=plotLabel[0],linewidth=2)
    ax[0].plot(tm/norm_time, At, '--r', label=plotLabel[1],linewidth=2)
    ax[0].grid(True)
    # ax[0].legend(loc='upper right',fancybox=True, shadow=True)
    ax[1].plot(tm/norm_time, Et0, 'g', label=plotLabel[0],linewidth=2)
    ax[1].plot(tm/norm_time, Et, '--r', label=plotLabel[1],linewidth=2)
    ax[1].grid(True)
    # ax[1].legend(loc='upper right',fancybox=True, shadow=True)

    plt.subplots_adjust(left=0.15,right=0.8,bottom=0.15)

    ax[1].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))
    ax[0].legend(loc='upper left',fancybox=True, shadow=True,bbox_to_anchor=(1., 1))

    ax[1].set_xlabel(r'$t$ (days)',fontweight='bold',fontsize=18)

    if (title==None):
        title = "Time Domain"
    fig.suptitle(title)

    if (ylabel==None):
        ax[0].set_ylabel(r'$A$',fontweight='bold',fontsize=18)
        ax[1].set_ylabel(r'$E$',fontweight='bold',fontsize=18)
    else:
        ax[0].set_ylabel(ylabel[0],fontweight='bold',fontsize=18)
        ax[1].set_ylabel(ylabel[1],fontweight='bold',fontsize=18)
    if (xlim != None):
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    if (ylim != None):
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)

    if save:
        if (picDir==None):
            picDir="../pictures/"
        plt.savefig(picDir+title+".jpg")
    # plt.gcf().subplots_adjust(bottom=0.2,left=0.15)
    plt.show()
    return(fig,ax)
