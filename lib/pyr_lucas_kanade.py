"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""
"""---------- Centre for Robotics ---------"""

import cv2
import numpy as np
from lib.rejection import inlier_dynamic, inlier_static



def optimize_x_y(q,sh,l,wz):
    # create window size
    w = wz

    # generate 2d array of points in the window size of x,y for all points
    range_x = [np.arange(val-w, val+w+1) for val in q[:,0]]
    range_y = [np.arange(val-w, val+w+1) for val in q[:,1]]
    x = np.minimum(np.array(range_x),sh[l,1]-1)

    # limit the maximum value for y to the size of the current array -1
    y = np.minimum(np.array(range_y),sh[l,0]-1)

    # limit the minimum points to atleast 0
    x[x<0] = 0
    y[y<0] = 0
    
    # create different combinations of the window sized element to index, e.g for x [1 1 2 2 3 3] for y [1 2 3 1 2 3]
    x = np.repeat(x,w,axis=1) # x,y is 2D array, x dimension is diffrent x,y y dimension are different points
    y = np.tile(y,w)
    x_ = x-1
    x_[x_< 1] = 0
    y_ = y - 1
    y_[y_<1] = 0
    return np.intp(x),np.intp(y),np.intp(x_),np.intp(y_)

def optimize_Ix_and_Iy(I_L,sh,l, x,y,x_,y_):
    #I_x = np.zeros((S[0],w**2), dtype=np.float32)
    #I_y = np.zeros((S[0],w**2), dtype=np.float32)
    #j = 0
    #print(np.shape(y),np.shape(np.minimum(x + 1, sh[l, 1] - 1)))
    I_x = (I_L[y, np.minimum(x + 1, sh[l, 1] - 1)] - I_L[y, x_ ]) / 2
    I_y = (I_L[np.minimum(y + 1, sh[l, 0] - 1), x] - I_L[y_, x,]) / 2
    
    return I_x, I_y

def optimized_dIk(I_L,J_L,x,y,v_k,g_L,l,sh):
    vy = (v_k[:,1]).reshape(len(v_k),1)
    vx = (v_k[:,0]).reshape(len(v_k),1)
    gy = (g_L[:,1]).reshape(len(g_L),1)
    gx = (g_L[:,0]).reshape(len(g_L),1)
    #print(np.shape(vy),np.shape(gy))
    k = np.intp(np.round(y+vy+gy))
    m = np.intp(np.round(x+vx+gx))
    k[k<1] = 0
    m[m<1] = 0
    k[k>sh[l,0]-1] = sh[l,0]-1
    m[m>sh[l,1]-1] = sh[l,1]-1
    
    dI_k = I_L[y,x] - J_L[k,m]
    #print("dI_k",dI_k)
    return dI_k




def lucas_pyramidal(img1_,img2_, number_featues,wz,level,k, inlier_threshold, static):
    img1 = np.array(cv2.cvtColor((img1_), cv2.COLOR_BGR2GRAY))
    img2 = np.array(cv2.cvtColor((img2_), cv2.COLOR_BGR2GRAY))
    I1 = np.array(img1)
    I2 = np.array(img2)
    S = np.shape(I1)
    #print(S)
    I_L = np.empty((S[0],S[1],level),dtype=np.float32)
    J_L = np.empty((S[0],S[1],level),dtype=np.float32)
    I_L[:,:,0] = I1
    J_L[:,:,0] = I2
    sh = np.empty((level,2),dtype=int)
    sh[0,:] = S
    
    # create image levels
    for i in range(1,level):
        sh[i,:] = np.shape(cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5))
        I_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
        J_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(J_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
    
    # get good features to track and convert them to an nx2 array
    features = cv2.goodFeaturesToTrack(I1,number_featues,0.01,10)
    q1 =[]
    '''feature = np.int0(features)
    for i in range(len(feature)):
        q1.append(([feature[i,0,0], feature[i,0,1]]))'''

    q2 = np.intp(np.array(features).reshape(len(features),-1))

    # Initial guess
    g_Lm = np.zeros((len(q2),2),dtype=np.float32)
    v_k_out = []

    '''g_Lm_0 = np.array(initial_guess)
    for ij in range(len(g_Lm)):
        g_Lm[ij,:] = g_Lm_0'''
    #print(initial_guess)
    # calculate the optical flow for all points at a time for each level    
    for l in range(level-1,-1,-1):
        q = np.intp(q2/2**l)
        
        # convert the points to a 2d array of the points and their window size combinations for indexing
        x,y,x_,y_ = optimize_x_y(q,sh,l,wz)

        # Calculate the derivatives of the intensity I_L
        Ix,Iy = optimize_Ix_and_Iy(I_L[:,:,l],sh,l, x,y,x_,y_)
        Ixy = Ix*Iy

        #form the spatial gradient matix of the points around x,y as an nx2x2
        I2x = Ix*Ix
        I2y = Iy*Iy
        Ix_ = np.sum(I2x, axis=1)
        Iy_ = np.sum(I2y, axis=1)
        Ixy_ = np.sum(Ixy, axis=1)
        a = np.dstack((Ix_,Ixy_,Ixy_,Iy_))
        G = a.reshape(len(Ix),2,2)
        G_ = np.linalg.pinv(G)
        # initialize the oprical flow for level k=0
        v_k = np.zeros((len(x),2),dtype=np.float32)
        
        for j in range(k):
            # get the image differnce
            dIk = optimized_dIk(I_L[:,:,l],J_L[:,:,l],x,y,v_k,g_Lm,l,sh)
            # create the image mismatch vector b
            dIkx = dIk*Ix
            dIky = dIk*Iy
            dIkx = np.sum(dIkx, axis=1)
            dIky = np.sum(dIky, axis=1)
            b = np.dstack((dIkx,dIky)).reshape(len(Ix),2,1)

            # optical flow LK
            n_k = np.matmul(G_,b)
            n_k = n_k.reshape(np.shape(v_k))

            # guess for the next iteration
            v_k += n_k
            #print(n_k)
        #print(n_k)
        #print(j-1)
        g_Lm = 2*(v_k+g_Lm)
        v_k_out.append(np.median(v_k,axis=0))
        #print(np.median(v_k, axis=0))
        #print(b)

    d = g_Lm/2
    #dm = np.median(d, axis=0)
    #print(dm)
    #q2, d, foes, n = brut_force(q2,d,S,3)
    if static:
        q2, d, foes = inlier_static(q2, d, S, inlier_threshold)
    else:
        q2, d, foes = inlier_dynamic(q2, d, S, inlier_threshold)
    #foes = 0
    #print(foes)
    q3 = q2 + d
    #print(np.median(d, axis=0))
    return q2, q3 , foes
