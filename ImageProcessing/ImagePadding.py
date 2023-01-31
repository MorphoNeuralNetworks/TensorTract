# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:14:02 2022

@author: aarias
"""

import numpy as np



def pad_3DImage(img3D, dz=0, dy=0, dx=0):
    img3D_pad = np.pad(img3D, ((dz, dz),(dy, dy), (dx,dx)))
    return img3D_pad

def pad_2DImage(img2D, dy, dx):
    img2D_pad = np.pad(img2D, ((dy, dy), (dx,dx)))
    return img2D_pad
    
def crop_3DImage(img, dz, dy, dx):
   sy, sx, sz = img.shape
   img_crop = img[0+dy:sy-dy, 0+dx:sx-dx, 0+dz:sz-dz]
   return  img_crop

def pad_test():
    img3D = np.ones((2,2,2))
 
    #Padding: z, y, x
    img3D_pad = np.pad(img3D, ((2,2), (0, 0), (0,0)))

    print()
    print('img3D:\n', img3D) 
    print('img3D_pad:\n', img3D_pad) 

# =============================================================================
# Test       
# =============================================================================
if __name__== '__main__':
    from IO.Image.ImageReader import read_Image
    from IO.Image.ImageWriter import save3Dimage_as3DStack
    from IO.Files.FileManager import createFolder
    from pathlib import Path

    # Read ImgIn
    pathFolder_ReadImage    = Path(r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\Control')
    pathFolder_ReadImage    = Path(r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\TH')
    # pathFolder_ReadImage    = Path(r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\NPY')
    
    img3D = read_Image(str(pathFolder_ReadImage), nThreads=1)
    
    #Padding
    img3D_pad = pad_3DImage(img3D, dz=0, dy=0, dx=100)
    
    print()
    print('dim: \n', img3D.shape)
    print('dim: \n', img3D_pad.shape)
    
    # #save
    # pathFolder = pathFolder_ReadImage.parent / (pathFolder_ReadImage.name + '_pad')    
    # createFolder(str(Path(pathFolder)), remove=False) #Create Folder
    # fileName = (pathFolder_ReadImage.name + '_pad.tif')    
    # save3Dimage_as3DStack(img3D_pad, pathFolder, fileName)

    
    #Save
    #Inve
    img3D = 255 - img3D
    #Padding
    img3D_pad = pad_3DImage(img3D, dz=0, dy=0, dx=100)
    pathFolder = pathFolder_ReadImage.parent / (pathFolder_ReadImage.name + '_pad_inv')    
    createFolder(str(Path(pathFolder)), remove=False) #Create Folder
    fileName = (pathFolder_ReadImage.name + '_pad_inv.tif')    
    save3Dimage_as3DStack(img3D_pad, pathFolder, fileName)
    

        
        
        
        
        