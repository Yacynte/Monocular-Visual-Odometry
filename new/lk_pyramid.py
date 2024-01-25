from pickletools import uint8
from new.PA2_2 import LucasKanade
import cv2
import numpy as np

def pyramid(img1,img2,level):
    I1 = np.array(img1)
    I2 = np.array(img2)
    S = np.shape(I1)

    pyramid1 = np.empty((S[0],S[1],level),dtype=np.uint8) 
    pyramid2 = np.empty((S[0],S[1],level),dtype=np.uint8) 
    pyramid1[:,:,0] = I1 			#since the lowest level is the original image
    pyramid2[:,:,0] = I2 			#since the lowest level is the original image
    sz = np.empty((level,2),dtype=int)
    sz[0,:] = np.shape(I1)
    
    
    #d = np.zeros((len(feature),2),dtype=float)
    #p = np.zeros((len(feature),2),dtype=float)
    for i in range(1,level):
        sz[i,:] = np.shape(cv2.pyrDown(pyramid1[0:sz[i-1,0],0:sz[i-1,1],i-1]))
        pyramid1[0:sz[i,0],0:sz[i,1],i] = cv2.pyrDown(pyramid1[0:sz[i-1,0],0:sz[i-1,1],i-1])
        pyramid2[0:sz[i,0],0:sz[i,1],i] = cv2.pyrDown(pyramid2[0:sz[i-1,0],0:sz[i-1,1],i-1])

    features = cv2.goodFeaturesToTrack(pyramid1[:,:,0], 10000, 0.01, 10)

    features = np.int0(features)
    feature = np.intp(np.round(np.divide(features, 2**(level-1))))
    g = np.zeros((len(feature),2),dtype=float)
    d, p = LucasKanade(I1,I2,features)
    #d1 , p1 = LucasKanade(pyramid1[0:sz[level-1,0],0:sz[level-1,1],level-1],pyramid2[0:sz[level-1,0],0:sz[level-1,1],level-1],features)
    #d, p = LucasKanade(pyramid1[0:sz[level-1,0],0:sz[level-1,1],level-1],pyramid2[0:sz[level-1,0],0:sz[level-1,1],level-1],feature)
    '''
    for i in range(level-2, 0, -1):
        fea = feature
        g = np.multiply(np.add(g,d),2)
        feature = np.multiply(feature,2)
        d, p = LucasKanade(pyramid1[0:sz[i,0],0:sz[i,1],i],pyramid2[0:sz[i,0],0:sz[i,1],i],feature)

        
    d = np.add(d,g)
    dept = np.add(p,d)
    '''
    return p, d