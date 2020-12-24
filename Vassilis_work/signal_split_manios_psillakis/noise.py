from scipy import optimize,ndimage
import numpy as np


def smoothing(X, smooth=5, transition=10, spike_width=7):
    """
    Only remove noise from low noiset signal area ’ s to
    maintain the intensity o f the spikes .
    Noise is removed with a gaussian filter inspectral dimension .
    In :
    X: datasetwith dimensions ( x*y , wavenumbers )
    Out :
    X_new: datasetwith dimensions ( x*y , wavenumbers )
    """
    grad= ndimage.gaussian_filter(X,(0,1),order=1) #gradient
    grad_abs=np.abs(grad)
    grad_abs_sm=ndimage.gaussian_filter(grad_abs,(0,5))
    mean_grad=np.mean(grad_abs,1)+1/np.std( grad_abs , 1)* 3
    spikes=((grad_abs_sm.T > mean_grad ).astype(float)).T
    spikes=np.round(ndimage.gaussian_filter(spikes,(0,spike_width)))
    spikes=ndimage.uniform_filter(spikes,(0,transition))
    return (1-spikes )*ndimage.gaussian_filter (X,(0,smooth))+ spikes*X
