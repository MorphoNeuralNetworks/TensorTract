# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 18:25:13 2020

@author: pc
"""

from scipy import signal
import numpy as np
import time
#==============================================================================
# 
#==============================================================================
def resample_Zdim(imgIn, z_thick, dissectionSize_iso, overlap_iso, t0=0):
    start = time.time()-t0
    #Forcing a 3D-Image to be Isotropic through a Resampling Operator    
    if z_thick>1.0:   
        zDim = imgIn.shape[2]*z_thick        
        Nz = (zDim-overlap_iso[0])/(dissectionSize_iso[0]-overlap_iso[0])
        Nz = int(np.round(Nz))
        zDim = Nz*(dissectionSize_iso[0]-overlap_iso[0]) + overlap_iso[0]    
        zDim = int(zDim)
        
        imgIn = signal.resample(imgIn, zDim, axis=2)
    
    stop = time.time()-t0
    return imgIn, start, stop

def resample_3DImage(imgIn, scannerSize_Out, t0=0, verbose=False):
    start = time.time() - t0
    if verbose==True:
        print()
        print('Start: resample_3DImage()')
        print('scannerSize_Out \n', scannerSize_Out)
        ny, nx, nz = imgIn.shape
        print('imgDimXYZ \n', [nx, ny, nz] )
        
    #Determine the Bit Depth
    dataType = imgIn.dtype.type
    bitDepth = 0
    if dataType==np.uint8:
        bitDepth = 8
    elif dataType==np.uint16:
        bitDepth = 16
    else:
        print()
        print('BitDepth not Found')
        print('dataType:', dataType)
        
    # print()    
    # print('bitDepth', bitDepth)
    # print('dataType:', dataType)
    
    # print()
    # print('Before: Resampled')
    # print('imgIn.min()', imgIn.min())
    # print('imgIn.max()', imgIn.max())
    
    #yxz
    imgIn = signal.resample(imgIn, scannerSize_Out[0], axis=1) # ??? IMP
    imgIn = signal.resample(imgIn, scannerSize_Out[1], axis=0) # ??? IMP
    imgIn = signal.resample(imgIn, scannerSize_Out[2], axis=2)
    
    # print()
    # print('After: Resampled')
    # print('imgIn.min()', imgIn.min())
    # print('imgIn.max()', imgIn.max())
    
    
    # if scannerSize_Out[2]!=0:
    #     imgIn = signal.resample(imgIn, scannerSize_Out[2], axis=2)
    
    # Recompute DataType
    imgIn[imgIn<0] = 0 
    imgIn = imgIn.astype(dataType)
    
    # print()
    # print('After: DataType Restorage')
    # print('imgIn.min()', imgIn.min())
    # print('imgIn.max()', imgIn.max())
    
    if verbose==True:
        ny, nx, nz = imgIn.shape
        print('imgDimXYZ \n', [nx, ny, nz] )
        print('Stop: resample_3DImage()')
        print()

    stop = time.time()-t0
    return imgIn, start, stop
        
if __name__== '__main__':
    pass