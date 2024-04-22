"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""
"""---------- Centre for Robotics ---------"""

import csv
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from integrate import integrate

def imu_position(imu_data, t):
        
        # Initialize velocity and position arrays
        imu_data = np.array(imu_data)
        velocity = imu_data[:,:3]
        acceleration = imu_data[:,3:]
        position = np.zeros(np.shape(acceleration))
        '''for i in range(1,len(acceleration)):
            velocity[i,:2] = acceleration[i,:2] * t[i] + velocity[i, :2]
            velocity[i, 2] = (acceleration[i,2]+9.81) * t[i]  + velocity[i, 2]'''

        for i in range(1, len(t)):
            position[i,0] = position[i-1,0] + velocity[i-1,0] * t[i]*2 - 0.5 * acceleration[i-1,0] * t[i]**2
            position[i,1] = position[i-1,1] - velocity[i-1,1] * t[i] - 0.5 * acceleration[i-1,1] * t[i]**2
            position[i,2] = position[i-1,2] + 0.5*(velocity[i-1,2] * t[i] + (acceleration[i-1,2] + 9.81) * t[i]**2)
        """# Plot results
        plt.figure()
        plt.plot(time_step, position[:,0])
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title('Estimated Position from Accelerometer Data')
        plt.grid(True)
        plt.show()
        print(nlfd)"""
        return position



def get_opt_gt(filename, timestamps):
    timestamp = []
    with open(timestamps, newline='\n', encoding='utf-8') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        k = 0
        for time_stamp in file:
            if k>0 :
                timestamp.append((time_stamp[1],time_stamp[2],time_stamp[6], time_stamp[7], time_stamp[8], time_stamp[9],time_stamp[10], time_stamp[11], time_stamp[12] ))
            k += 1
    timestamp = np.array(timestamp, dtype=np.float64)
    timestamp[:,-2] = timestamp[:,-2] + 981
 
    with open(filename, newline='\n', encoding='utf-8') as csvfile:
        rec = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        j = 0
        k = 0
        poses = []
        prev_row = np.zeros((6))
        for row in rec:
            if i>=7 and j < len(timestamp):
                m = np.abs(np.float64(row[1]) - np.float64(timestamp[j,0]))
                if (np.abs(np.float64(row[1]) - np.float64(timestamp[j,0])) < 0.01) and (not ('' in row[2:8]) ):
                    row = np.array(row[2:8])                  
                    row = np.float32(row)
                    x, y, z, t_x, t_y, t_z = row
                    # convert angles to rotation vector
                    optitrak__rotation = np.array([np.radians(x), np.radians(y), np.radians(z)])

                    # Assume you have translation vector in AirSim
                    optitrak_translation_vector = [t_x, t_y, t_z]
        
                    # Convert rotation vector to rotation matrix
                    
                    # rotation_matrix_opencv = Rotation.from_euler('xyz', optitrak__rotation).as_matrix() 
                    rotation_matrix_opencv, _ = cv2.Rodrigues(optitrak__rotation)

                    # The translation vector remains the same
                    translation_vector_opencv = np.array(optitrak_translation_vector)

                    # Stack R and t together to form the transformation matrix
                    R = np.array([[0,0,1],[0,1,0],[1,0,0]])
                    transformation_matrix_opencv = np.eye(4)
                    transformation_matrix_opencv[:3, :3] = rotation_matrix_opencv
                    transformation_matrix_opencv[:3, 3] = translation_vector_opencv

                    transformation_matrix_opencv[:3,:] = np.matmul(R,transformation_matrix_opencv[:3,:])                    
                    poses.append(transformation_matrix_opencv)
                    if len(poses) == 6:
                        print(optitrak__rotation, rotation_matrix_opencv)
                    
                    j += 1
            i +=1

    poses_ = np.array(poses)  
    # print(poses_[5])

    # Integrate acceleration from IMU to get the distances
    dist = integrate((timestamp[:,2:8])/100, timestamp[:,0],poses_)
    
   
    return poses, dist 

def get_airsim_gt(filename):

    def quaternion_to_euler(quaternion):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        :param quaternion: numpy array representing the quaternion [w, x, y, z]
        :return: numpy array containing Euler angles [roll, pitch, yaw] in radians
        """
        w, x, y, z = quaternion
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use +/-90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


    with open(filename,newline='\n',encoding='utf-8') as txtfile:
        rec = csv.reader(txtfile, delimiter='\t', quotechar='|')
        i = 0
        poses = []
        linear_acceleration = []
        timestamp = []
        for txtlist in rec:
            if i>0:
                pos = np.array(txtlist[2:9])
                # acc = np.float32(np.array(txtlist[12:15])) + np.float32(np.array([0,0,9.81]))
                times = float(txtlist[1])
                timestamp.append(times)
                pose_ = np.float32(pos)
                t_x,t_y,t_z,w,x,y,z = pose_
                # Assume you have rotation represented as a quaternion in AirSim
                airsim_rotation_quaternion = [w, x, y, z]

                # Assume you have translation vector in AirSim
                airsim_translation_vector = [t_x, t_y, t_z]

                # Convert quaternion to Euler angles
                rotation_angles = quaternion_to_euler(airsim_rotation_quaternion)
                # Convert Euler to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_angles)
                # rotation_matrix_opencv, _ = cv2.Rodrigues(np.array(airsim_rotation_quaternion[1:]))  
                # Convert quaternion to axis-angle representation and then to rotation matrix

                # The translation vector remains the same
                translation_vector_opencv = np.array(airsim_translation_vector)

                # Stack R and t together to form the transformation matrix
                R = np.array([[1,0,0],[0,0,1],[0,1,0]])
                transformation_matrix_opencv = np.eye(4)
                transformation_matrix_opencv[:3, :3] = rotation_matrix
                transformation_matrix_opencv[:3, 3] = translation_vector_opencv
                R1 = Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
                # transformation_matrix_opencv[:3,:] = np.matmul(R1,transformation_matrix_opencv[:3,:])
                poses.append(transformation_matrix_opencv)
                # acc = (R@acc.reshape(3,1)).reshape(3)
                # linear_acceleration.append(acc)
            i += 1
            
    # print(poses)
    R1 = Rotation.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()

    linear_acceleration = np.array(linear_acceleration) 
    timestamp = np.array(timestamp)
    timestamp = (timestamp - timestamp[0])/1000
   
    poses_ = np.array(poses)
    # position_of_imu = imu_position(linear_acceleration, timestamp)
    poses_[:3,:3] = R1@poses_[:3,:3]

    # dist = integrate_airsim(linear_acceleration, timestamp, poses_)
    dist = []
    for i in range(1,len(poses_)):
        # d_vo.append(np.linalg.norm(vo[i]-vo[i-1]))
        dist.append(np.linalg.norm(poses_[i,:3,3]-poses_[i-1,:3,3]))

    return poses, dist
