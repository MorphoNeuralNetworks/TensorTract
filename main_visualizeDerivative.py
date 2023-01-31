# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:14 2020

@author: pc
"""

    


import numpy as np
from scipy import signal
import time


#from ImageFilters import get_DxGaussian, get_Gaussian
from ImageProcessing.ImageFilters import get_DxGaussian, get_Gaussian
        
import os
import psutil
import itertools


from ImageProcessing.ImageResampling import resample_3DImage

#GUI 
from PyQt5 import QtWidgets, QtCore 

#Custom GUI Design
from qt_main_VisPy import Ui_MainWindow 

#Solve Issues
from multiprocessing import freeze_support

#System
import sys
import os


#GPU Scientific Visualization 
import vispy
from vispy import scene
# from vispy.visuals.transforms import STTransform
from vispy.visuals.transforms import STTransform, MatrixTransform, ChainTransform
import vispy.io as io

from GUI_VisPy.VisPyManager import (embed_visViewBoxIntoQtWidget,
                                    add_VisCamToViewBox,
                                    add_visDisplay,
                                    add_visVolObjectToView,
                                    add_visBoxObjectToView,
                                    add_visCubeObjectToView,
                                    add_visGridObjectToView,        
                                    add_visDotsObjectToView,
                                    add_visLineObjectToView,
                                    
                                    update_VisCam,
                                    update_visVol,
                                    
                                    plot_visBox,
                                    plot_visXYZAxis, 
                                    plot_visDots,
                                    # plot_visOrthogonal,
                                    
                                    
                                    update_visDisplay,
                                    update_visOrthoView,
                                    update_visBox, 
                                    update_visCube,
                                    update_visGrid,
                                    update_visPlane,
                                    update_visDots,
                                    update_visLine,
                                    
                                    remove_visObjects,
                                    # add_vis2DDisplay,
                                    # update_vis2DDisplay,
)

#GUI Class
class mywindow(QtWidgets.QMainWindow): 
    def __init__(self): 
        super(mywindow, self).__init__()     
        self.ui = Ui_MainWindow()        
        self.ui.setupUi(self)
        
        #Init GUI-PyQT5 
        # self.init()
        # self.init_GUI_Events()
        # self.init_GUI_Var()
        
        # Init: GUI-VisPy 
        self.init_VisCanvas()
        self.show()
        
        
        # Simulate: users Clicks 
        # self.byPass_GUI_Var()
        # self.init_JSON() 
        
    def init_VisCanvas(self):
        from pathlib import Path
        from IO.Image.ImageReader import read_Image
        
        
        visOrtho_XYZ = add_visDisplay(self.ui.visWidget_3D, 'Turntable')
        visOrtho_XY = add_visDisplay(self.ui.visWidget_XY, 'PanZoom')
        visOrtho_YZ = add_visDisplay(self.ui.visWidget_YZ, 'PanZoom')
        visOrtho_XZ = add_visDisplay(self.ui.visWidget_XZ, 'PanZoom')
        
        
        pathFolder_ReadImage  = Path().absolute() / "Examples\MiniTest_v0\MiniTest_Tract_IsoCrop.tif"
        # pathFolder_ReadImage  = Path().absolute() / "Examples\MiniTest_v0\Dxx.tif"
        # pathFolder_ReadImage  = Path().absolute() / "Examples\MiniTest_v0\Dyy.tif"
        pathFolder_ReadImage  = Path().absolute() / "Examples\MiniTest_v0\Dxy.tif"
        img3D = read_Image(pathFolder_ReadImage, nThreads=1)
        
        # Visualize
        update_visOrthoView(img3D, visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ)


if __name__ == '__main__':
    #Prevents issues with pyinstaller
    freeze_support()
    
    # checks if QApplication already exists 
    app = QtWidgets.QApplication.instance() 
    if not app: 
        # create QApplication if it doesnt exist 
          app = QtWidgets.QApplication([])   
    # to cause the QApplication to be deleted later
    app.aboutToQuit.connect(app.deleteLater)    
    
    application = mywindow()     
    application.show()     
    sys.exit(app.exec_())
    

