# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:14 2020

@author: pc
"""

import numpy as np
from ImageProcessing.ImageResampling import resample_3DImage


    
if __name__ == '__main__':
    
    # Path Management
    from pathlib import Path
    from IO.Image.ImageReader import read_Image
    from IO.Files.FileManager import createFolder
    from IO.Image.ImageWriter import save3Dimage_as3DStack
    from ImageProcessing.Tensor import compute_Tensor
    from ImageProcessing.ImagePadding import crop_3DImage

    # Read Image
    pathFolder_ReadImage    = Path().absolute() / "Examples\MiniTest_v0\MiniTest_Tract_Ani.tif"
    img3D = read_Image(pathFolder_ReadImage, nThreads=1)  
    
    #Resample
    voxelSize_In_um  = np.asarray([0.45, 0.45, 2.00])
    voxelSize_Out_um = 2*np.asarray([1.0, 1.0, 1.0]) 
    
    # Get the Out/In Ratio 
    r_um = voxelSize_Out_um/voxelSize_In_um
    r_px = 1/r_um
    
    # Compute the Output Dimensions
    imgDimYXZ_In = np.array(img3D.shape)  
    imgDimXYZ_In = imgDimYXZ_In[[1,0,2]]
    imgDimXYZ_Out = np.round(imgDimXYZ_In*r_px).astype(int)
    
    print()
    print('Size')
    print('imgDimXYZ_In: ', imgDimXYZ_In) 
    print('imgDimXYZ_Out:', imgDimXYZ_Out) 
    print('r_um:', r_um)

    
    # Resample Image
    if (r_um[0]!=1)|(r_um[1]!=1)|(r_um[2]!=1): 
        print()
        print('Compute Resampling...')
        [img3D, start, stop]  = resample_3DImage(img3D, imgDimXYZ_Out)
        
        #Save Resampled Image as a 3D tiff
        pathFolder =  pathFolder_ReadImage.parent
        createFolder(str(pathFolder), remove=False) # !!! Fails    
        # createFolder(str(pathFolder), remove=True) #!!!
        fileName   = "MiniTest_Tract_Iso"
        save3Dimage_as3DStack(img3D, pathFolder, fileName)
        
    #Compute Partial Derivatives
    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = compute_Tensor(img3D, s=5, k=1.0)
    # [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = compute_Tensor(img3D, s=2, k=1.0)
    
    image_padding = 10 # pixels
    img3D = crop_3DImage(img3D, dz=image_padding, dy=image_padding, dx=image_padding)
    Dxx = crop_3DImage(Dxx, dz=image_padding, dy=image_padding, dx=image_padding)
    Dyy = crop_3DImage(Dyy, dz=image_padding, dy=image_padding, dx=image_padding)
    Dxy = crop_3DImage(Dxy, dz=image_padding, dy=image_padding, dx=image_padding)

    
    #Save Resampled Image as a 3D tiff
    pathFolder =  Path().absolute().parent
    pathFolder =  pathFolder_ReadImage.parent
    createFolder(str(pathFolder), remove=False) # !!! Fails    
    fileName   = "MiniTest_Tract_IsoCrop"
    save3Dimage_as3DStack(img3D, pathFolder, fileName)
    
    #Save Resampled Image as a 3D tiff
    pathFolder =  Path().absolute().parent
    pathFolder =  pathFolder_ReadImage.parent
    createFolder(str(pathFolder), remove=False) # !!! Fails    
    fileName   = "Dxx"
    save3Dimage_as3DStack(Dxx, pathFolder, fileName)
    
    pathFolder =  Path().absolute().parent
    pathFolder =  pathFolder_ReadImage.parent
    createFolder(str(pathFolder), remove=False) # !!! Fails    
    fileName   = "Dyy"
    save3Dimage_as3DStack(Dyy, pathFolder, fileName)
    
    pathFolder =  Path().absolute().parent
    pathFolder =  pathFolder_ReadImage.parent
    createFolder(str(pathFolder), remove=False) # !!! Fails    
    fileName   = "Dxy"
    save3Dimage_as3DStack(Dxy, pathFolder, fileName) 
 

