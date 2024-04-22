"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""

import numpy as np
from lib.ransac import estimate_foe

def inlier_static(q2,d,S,inlier_threshold):
    foe, inlie = estimate_foe(d, inlier_threshold )
    #inlie = np.array(inlie)
    q2 = q2[inlie]
    d = d[inlie]
    a = np.sqrt(d[:,0]**2 + d[:,1]**2)
    dm = np.median(a, axis=0)
    dm = np.linalg.norm(dm)
    #print(dm)
    drr = np.where(a < 2*dm) #& (a > 0.2*dm))
    dr = d[drr]
    qr = q2[drr]


    """alp = np.arctan2(dr[:,1],dr[:,0])*180/np.pi
    alp_me = np.median(alp, axis=0)
    #foe = dm
    # print(alp_me)
    drr_alp = np.where((np.abs(alp) < np.abs(alp_me)+10) & (np.abs(alp) > np.abs(alp_me)-10) )
    # print(np.shape(d))
    dr_alp = dr[drr_alp]
    qr_alp = qr[drr_alp]"""

    return qr, dr, foe+np.array([S[1]/2,S[0]/2])

def inlier_dynamic(q2,d,S,inlier_threshold):
    n = 3
    S = np.array(S)
    u = np.zeros((S[0],S[1],2))
    u[np.reshape(q2[:,1],-1),np.reshape(q2[:,0],-1)] = d
    S = np.intp(S/n)
    foes = np.zeros((n*n,2))
    i3 = 0
    #all_inliners = [] 
    dd = [] #np.empty((1,2),dtype=np.float32)
    qd = [] #ddr.copy()
    for j in range(0,n):
        for i in range(0,n):
            i3 += 1
            k = q2[np.where(np.all(np.array([i*S[1],j*S[0]]) <= q2,axis=1) & np.all(np.array([(i+1)*S[1],(j+1)*S[0]])>q2,axis=1)),:]
            k = k[0,:,:]
            d_rob = u[np.reshape(k[:,1],-1),np.reshape(k[:,0],-1)]
            if np.shape(d_rob)[0] <= 2:
                # print("break")
                continue
            
            foe, inliers = estimate_foe(d_rob, inlier_threshold)
            foes[i3-1,:] = np.array(foe) + np.array([(i+0.5)*S[1],(j+0.5)*S[0]])
            q_rob = k[inliers]
            d_rob = d_rob[inliers]
            if np.shape(d_rob)[0] <= 2:
                # print("break")
                continue
            a = np.sqrt(d_rob[:,0]**2 + d_rob[:,1]**2)
            dm = np.median(a)
            dm = np.linalg.norm(dm)
            # print(dm)
            drr = np.where(a < 2*dm) #& (a > 0.2*dm))
            dr = d_rob[drr]
            qr = q_rob[drr]
            for di, qi in zip(dr, qr):
                dd.append(di)
                qd.append(qi)

            '''inliers = np.array(inliers) #+ np.array([(j+1)*S[0],(i+1)*S[1]])
            #print(i+j)
            if i+j == 0:
                #all_inliers = inliers
                ddr = d_rob[inliers]
                kd = k[inliers]
            else:
                #all_inliers = all_inliers + inliers
                ddr = np.concatenate((ddr,d_rob[inliers]),axis=0)
                kd = np.concatenate((kd,k[inliers]),axis=0)'''
            #print(type(all_inliers))

    #print(foes)
    # kd_, ddr_, foes_, n_ = brut_force2(kd, ddr, s, n)
    return np.array(qd), np.array(dd), foes
    #return kd, ddr , foes, n


def brut_force2(q2,d,S,n):
    S = np.array(S)
    s = S.copy()
    u = np.zeros((S[0],S[1],2))
    # v = np.flip(q2,1)
    q = q2.copy()
    d2 = d.copy()
    #print(np.reshape(q2[:,1],-1))
    u[np.reshape(q2[:,1],-1),np.reshape(q2[:,0],-1)] = d
    S = np.intp(S/n)
    
    foes = np.zeros((n*n,2))
    i3 = 0
    #all_inliners = [] 
    ddr = np.empty((1,2),dtype=np.float32)
    kd = ddr.copy()
    for j in range(0,n):
        for i in range(0,n):
            i3 += 1
            k = q2[np.where(np.all(np.array([i*S[1],j*S[0]]) <= q2,axis=1) & np.all(np.array([(i+1)*S[1],(j+1)*S[0]])>q2,axis=1)),:]
            k = k[0,:,:]
            d_rob = u[np.reshape(k[:,1],-1),np.reshape(k[:,0],-1)]
            if d_rob.size < 2:
                # print("break")
                continue
            
            foe, inliers = estimate_foe(d_rob)
            foes[i3-1,:] = np.array(foe) + np.array([(i+0.5)*S[1],(j+0.5)*S[0]])
            inliers = np.array(inliers) #+ np.array([(j+1)*S[0],(i+1)*S[1]])
            #print(i+j)
            if i+j == 0:
                #all_inliers = inliers
                ddr = d_rob[inliers]
                kd = k[inliers]
            else:
                #all_inliers = all_inliers + inliers
                ddr = np.concatenate((ddr,d_rob[inliers]),axis=0)
                kd = np.concatenate((kd,k[inliers]),axis=0)
            #print(type(all_inliers))
            a = np.sqrt(d_rob[:,0]**2 + d_rob[:,1]**2)
            dm = np.median(a, axis=0)
            #print(dm)
            drr = np.where((a > 1.5*dm) | (a < 0.2*dm))
            dr = d_rob[drr]
            qr = k[drr]

            dr = dr.tolist()
            qr = qr.tolist()
            d = d.tolist()
            q2 = q2.tolist()

            for i2,j2 in zip(dr,qr):
                d.remove(i2)
                q2.remove(j2)
            d = np.array(d)
            q2 = np.array(q2)

               
    for j in range(n):
        for i in range(n):
            k = q2[np.where(np.all(np.array([i*S[1],j*S[0]]) <= q2,axis=1) & np.all(np.array([(i+1)*S[1],(j+1)*S[0]])>q2,axis=1)),:]
            k = k[0,:,:]
            #print(k)
            d_rob = u[np.reshape(k[:,1],-1),np.reshape(k[:,0],-1)]
            if d_rob.size < 2:
                continue

            alp = np.arctan2(d_rob[:,1],d_rob[:,0])*180/np.pi
            alp_me = np.median(alp, axis=0)
            # print(alp_me)
            drr_alp = np.where((alp > alp_me+7.5) | (alp < alp_me-7.5) )
            # print(np.shape(d))
            dr_alp = d_rob[drr_alp]
            qr_alp = k[drr_alp]

            dr_alp = dr_alp.tolist()
            qr_alp = qr_alp.tolist()

            d = d.tolist()
            q2 = q2.tolist()
            for i2,j2 in zip(dr_alp,qr_alp):
                d.remove(i2)
                q2.remove(j2)
            
            d_ = np.array(d)
            q2_ = np.array(q2)

    #kd_, ddr_, foes_, n_ = brut_force(q2, d, s, n)    
    #print(foes)
    #return kd_, ddr_, foes_, n_
    return q2, d , foes