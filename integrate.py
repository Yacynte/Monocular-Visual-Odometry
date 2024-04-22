"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""
"""---------- Centre for Robotics ---------"""

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# Sample data for time and acceleration
# t = np.linspace(0, 10, 100)  # Time array from 0 to 10 seconds
# a = np.sin(t) + 1  # Example acceleration data (can be replaced with your dataset)

def integrate(a,t,gt_path_3d):
    # Numerical integration to calculate velocity
    vx = cumtrapz(a[:,3] + a[:,4], t, initial=0) #+ a[:,0]
    vy = cumtrapz(a[:,4] - a[:,4], t, initial=0) #+ a[:,1]
    vz = cumtrapz(-a[:,5] - a[:,4], t, initial=0) #+ a[:,2]


    # return dx, dy, dz
    # Numerical integration to calculate displacement
    dx = cumtrapz(vx, t, initial=0)
    dy = cumtrapz(vy, t, initial=0)
    dz = cumtrapz(vz, t, initial=0)
    vo = np.zeros((len(dx),3))
    vo[:,0] = dx #(dx+dy)#/10
    vo[:,1] = dy
    vo[:,2] = dz #-dy-dz
    d_vo = []
    d_gt = []
    for i in range(1,len(vo)):
        d_vo.append(np.linalg.norm(vo[i]-vo[i-1]))
        d_gt.append(np.linalg.norm(gt_path_3d[i,:3,3]-gt_path_3d[i-1,:3,3]))

    print(np.sum(d_vo), np.sum(d_gt))

    return d_vo
    # Plotting acceleration, velocity, and displacement
    plt.figure(figsize=(10, 6))
    plt.plot(t[1:], d_vo, label='Displacement vo (m)')
    plt.plot(t[1:], d_gt, label='Displacement gt (m)')
    # plt.plot(t, -dz-dy, label='Displacement Z (m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Acceleration, Velocity, and Displacement')
    plt.legend()
    plt.grid(True)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(dx+dy+gt_path_3d[0,2,3],dy-dy+gt_path_3d[0,0,3],-dz-dy+gt_path_3d[0,1,3],label='Visual Odometry')
    ax.plot(gt_path_3d[:,2,3],gt_path_3d[:,0,3],gt_path_3d[:,1,3],label='Ground Truth')
    # ax.plot(gt_path_3d[:,2,3],gt_path_3d[:,0,3],gt_path_3d[:,1,3],label='Visual Odometry')
    # ax.plot(gt_path_3d[:,2,3],gt_path_3d[:,0,3],gt_path_3d[:,1,3],label='Ground Truth')
    plt.grid()
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    plt.show()

    return dx, dy, dz



