from   scipy import ndimage
import numpy as np
import numpy.polynomial.polynomial as poly


def split_Raman_photoluminescence(X , wavelength):
    a = X
    border = 50
    # remove the top o f the s p i k e s from data ,
    # by us ing a Gaussian smoothing filter
    for _ in range (5) :
        a[:,border]=X[:,border]
        a[:, -border ] = X[ : , - border ]
        a1=ndimage.gaussian_filter(a,(0,30),mode='nearest')
        a=np.min([a,a1],axis=0)
    # remove the spikes from data , by using a polynominal fit
    for _ in range (5):
        a[:, border] = X[:, border]
        a[:, -border]=X[:, - border]
        z= poly.polyfit(wavelength[:: 5],a[:,:: 5].T, 5)
        a1= poly.polyval(wavelength, z)
        a=np.min([a, a1], axis=0)
    # smooth the curve the data ,
    # ( to remove remnants of noi s e in the photolumine s c enc e s i g n a l )
    for _ in range(10):
        a[:, 1] = X[: ,1]
        a[:, -1]=X[:, -1]
        a = ndimage.gaussian_filter(a,(0,10), mode='nearest')
    # make the Raman s i g n a l non􀀀ne gative ,
    # ( to remove remnants of noi s e in the Raman s i g n a l )

    return (X - a).clip(min=0), a


