from operator import le
import cv2
import numpy as np
import math


def py_lucas(img1,img2,level,wz):
    I1 = np.array(img1)
    I2 = np.array(img2)
    S = np.shape(I1)
    I_L = np.empty((S[0],S[1],level),dtype=np.float32)
    J_L = np.empty((S[0],S[1],level),dtype=np.float32)
    I_L[:,:,0] = I1
    J_L[:,:,0] = I2
    sh = np.empty((level,2),dtype=int)
    sh[0,:] = S
    a = S
    for i in range(1,level):
        a = [a[0]/2,a[1]/2]
        #print(a)
        sh[i,:] = np.shape(cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5))
        #print(sh)
        I_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(I_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
        J_L[0:sh[i,0],0:sh[i,1],i] = cv2.resize(J_L[0:sh[i-1,0],0:sh[i-1,1],i-1], None, fx = 0.5, fy = 0.5)
    '''
    for i in range(1,level):
        a = np.shape(cv2.pyrDown(I_L[0:int(S[0]/2**(i-1)),0:int(S[1]/2**(i-1)),i-1]))
        I_L[0:int(a[0]),0:int(a[1]),i] = cv2.pyrDown(I_L[0:int(S[0]/2**(i-1)),0:int(S[1]/2**(i-1)),i-1])
        J_L[0:int(a[0]),0:int(a[1]),i] = cv2.pyrDown(J_L[0:int(S[0]/2**(i-1)),0:int(S[1]/2**(i-1)),i-1])
        sh[i,:] = a
    '''
    features = cv2.goodFeaturesToTrack(I1,10000,0.01,10)
    q1 =[]
    feature = np.int0(features)
    #for i in range(int(len(feature)/2)):
    for i in range(100):
        q1.append(([feature[i,0,0], feature[i,0,1]]))
    q2 = np.array(q1)
    #print(I1[0:20,0:10])
    #print(I_L[0:55,0:12,4])
    #print(J_L[0:55,0:12,4])
    def Ix(x,y,l):
        if x+1 >= sh[l,1] -1:
            n = sh[l,1]-1
        else:
            n = x+1
        #print("in Ix, ", I_L[y,x+1,l],I_L[y,x-1,l])
        I_x = (I_L[y,n,l] - I_L[y,x-1,l])/2
        return I_x
    
    def Iy(x,y,l):
        if y+1 >= sh[l,0] -1:
            n = sh[l,0]-1
        else:
            n = y+1
        I_y = (I_L[n,x,l] - I_L[y-1,x,l])/2
        return I_y
    
    def dIk(x,y,v_k,g_L,l,s):
        k = int(round(y+v_k[1]+g_L[1]))
        m = int(round(x+v_k[0]+g_L[0]))
        if m<=0:
            m=0
        if k<=0:
            k=0
        if m >= sh[l,1] -1:
            m = sh[l,1]-1
        if k >= sh[l,0] -1:
            k = sh[l,0]-1
        #if l == 3:
        #    print("x,y",(x,y))
        #print("m,k",(m,k))
        dI_k = I_L[y,x,l] - J_L[k,m,l]
        #print("dI_k",dI_k)
        return dI_k
    
    p = []
    q = 0

    for px_,py_ in q2:
        g_Lm = np.array(([0,0]),dtype=np.float32)
        
        for l in range(level-1,-1,-1):
            px = int(px_/2**l)
            py = int(py_/2**l)
            #print(l)
            #print("px =",px,px_)
            #print(g_Lm)
            #print(G)
            #print("px,py",px,py)
            G = np.asmatrix(np.array(([[0,0],[0,0]]),dtype=np.float32))
            v = np.array(([0,0]),dtype=np.float32)
            for y in range(py-wz,py+wz+1,1):
                for x in range(px-wz,px+wz+1,1):
                    if x<=0:
                        x = 1
                    if y<=0:
                        y = 1
                    if x >= sh[l,1] -1:
                        x = sh[l,1]-1
                    if y >= sh[l,0] -1:
                        y = sh[l,0]-1
                    #print("x,y",(x,y))
                    G += np.asmatrix(np.array(([[Ix(x,y,l)**2,Ix(x,y,l)*Iy(x,y,l)],[Ix(x,y,l)*Iy(x,y,l),Iy(x,y,l)**2]]),dtype=np.float32))
            #n_k = np.array([1,1])
            #while ((np.abs(n_k)) > np.array([0.25, 0.25])).all():
            for k in range(1,7):
                b_k = np.asmatrix(np.array(([[0],[0]]),dtype=np.float32))
                for y in range(py-wz,py+wz+1,1):
                    for x in range(px-wz,px+wz+1,1):
                        if x<=0:
                            x = 1
                        if y<=0:
                            y = 1
                        if x >= sh[l,1] -1:
                            x = sh[l,1]-1
                        if y >= sh[l,0] -1:
                            y = sh[l,0]-1
                        #print("x,y,m,k",(x,y,int(round(v[0]+g_Lm[0]+x)),int(round(y+v[1]+g_Lm[1]))))
                        a = Ix(x,y,l)
                        b = Iy(x,y,l)
                        c = dIk(x,y,v,g_Lm,l,sh)
                        #print("Ix,Iy,dIk ",a,b,c)
                        b_k += np.asmatrix(np.array(([[dIk(x,y,v,g_Lm,l,sh)*Ix(x,y,l)],[dIk(x,y,v,g_Lm,l,sh)*Iy(x,y,l)]]),dtype=np.float32))
                #print(G)
                #print(b_k)
                #n_k = np.dot(np.linalg.pinv(np.array(np.matrix(G))),np.array(np.matrix(b_k)))
                n_k = np.dot(np.linalg.pinv(G),b_k)
                n_k = np.reshape(np.array(n_k),(2,))
                #if l == level-1:
                #print("n_k",n_k)
                v += n_k
            #print("v ",v)

            g_Lm = 2*(v+g_Lm)
            #print("gl",g_Lm)
            
        d = g_Lm/2
        #print("dl ",d_L)
        #print("gl ",g_Lm)
        p_ = np.array(([px,py]),dtype=np.float32)+d
        p.append(p_)

    q3 = np.array(p)
    return q2, q3

            