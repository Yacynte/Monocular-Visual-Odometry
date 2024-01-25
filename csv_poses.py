"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wueryburg-Schweinfurt ---------"""

import csv
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def euler_from_quaternion(x, y, z, w):
        
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        """x_e_.append(float(roll_x))
        y_e_.append(float(pitch_y))
        z_e_.append(float(yaw_z))

        return roll_x, pitch_y, yaw_z # in radians"""
        rx,ry,rz = roll_x, pitch_y, yaw_z
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]]) # base => nav  (level oxts => rotated oxts)
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) # base => nav  (level oxts => rotated oxts)
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]]) # base => nav  (level oxts => rotated oxts)
        R  = np.matmul(np.matmul(Rz, Ry), Rx)
        return R



def get_opt_gt(filename, timestamps):
    timestamp = []
    #print(timestamps)
    with open(timestamps, newline='\n', encoding='utf-8') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        k = 0
        for time_stamp in file:
            #print(time_stamp)
            if k>14:
                #print(time_stamp)
                timestamp.append((time_stamp[0],np.float64(time_stamp[1])))#, np.float64(time_stamp[8]), np.float64(time_stamp[9]), np.float64(time_stamp[9])))
            k += 1
    timestamp = np.array(timestamp)
    #print(np.shape(timestamp))
    with open(filename, newline='\n', encoding='utf-8') as csvfile:
        rec = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        j = 0
        k = 0
        poses = []
        prev_row = np.zeros((6))
        for row in rec:
            if i>=7 and j < len(timestamp):
                #print(row[1])
                #print(gfia)
                if (np.float64(row[1]) - np.float64(timestamp[j,1]) > 0.0000000001) and (not ('' in row[2:8]) ):
                    #print(timestamp[j,0])
                    """if ('' in row[2:8]):
                        row = prev_row + d"""
                    row = np.array(row[2:8])
                    
                    row = np.float32(row)
                    d = row -prev_row
                    
                    x, y, z, t_x, t_y, t_z = row
                    # Assume you have rotation represented as a quaternion in AirSim
                    optitrak__rotation = np.array([np.radians(x), np.radians(y), np.radians(z)])
                    #optitrak__rotation = np.array([x,y,z])
                    
                    #_quaternion = [w, x, y, z]

                    # Assume you have translation vector in AirSim
                    optitrak_translation_vector = [t_x, t_y, t_z]

                    # Convert quaternion to rotation matrix using OpenCV
                    
                    rotation_matrix_opencv = Rotation.from_euler('xyz', optitrak__rotation).as_matrix() 
                    #cv2.Rodrigues(np.array(optitrak__rotation))[0]
                    # Convert quaternion to axis-angle representation and then to rotation matrix

                    # The translation vector remains the same
                    translation_vector_opencv = np.array(optitrak_translation_vector)

                    # Stack R and t together to form the transformation matrix
                    R = np.array([[0,1,0],[1,0,0],[0,0,1]])
                    transformation_matrix_opencv = np.eye(4)
                    transformation_matrix_opencv[:3, :3] = rotation_matrix_opencv
                    transformation_matrix_opencv[:3, 3] = translation_vector_opencv

                    #transformation_matrix_opencv[:3,:] = np.matmul(R,transformation_matrix_opencv[:3,:])
                    """if j == 0:
                        print( optitrak_translation_vector)"""
                    
                    poses.append(transformation_matrix_opencv)
                    prev_row = np.copy(row)
                    
                    j += 1
            i +=1
        
    #poses = np.array(poses)
    #print(len(poseses))
    #print(poseses[0][:])
    #plt.plot(poses[:,0,3],poses[:,2,3])
    #plt.show()
    return poses#, np.float64(timestamp[:,1]), np.float64(timestamp[:,2:5])

def get_airsim_gt(filename):

    with open(filename,newline='\n',encoding='utf-8') as txtfile:
        rec = csv.reader(txtfile, delimiter='\t', quotechar='|')
        i = 0
        poses = []
        for txtlist in rec:
            if i>0:
                pos = np.array(txtlist[2:9])
                pose_ = np.float32(pos)
                t_x,t_y,t_z,w,x,y,z = pose_
                """R = euler_from_quaternion(pose_[4],pose_[5],pose_[6],pose_[3])
                pose_air = np.eye(4,4)
                pose_air[0:3,0:3] = R
                pose_air[0,3], pose_air[1,3], pose_air[2,3] = pose_[0],pose_[1],pose_[2]
                pose_cam = np.eye(4,4)
                pose_cam[0:3,:] = np.array([pose_air[0,:],-pose_air[2,:],pose_air[1,:]])

                pose_cam.tolist()
                poses.append(pose_cam)
                #print(pose_cam)
                if i==10:
                    #print(pose_air)
                    #print(R)
                    print(poses[0],poses[1],poses[2],poses[3])
                """
                # Assume you have rotation represented as a quaternion in AirSim
                airsim_rotation_quaternion = [w, x, y, z]

                # Assume you have translation vector in AirSim
                airsim_translation_vector = [t_x, t_y, t_z]

                # Convert quaternion to rotation matrix using OpenCV
                rotation_matrix_opencv, _ = cv2.Rodrigues(np.array(airsim_rotation_quaternion[1:]))  
                # Convert quaternion to axis-angle representation and then to rotation matrix

                # The translation vector remains the same
                translation_vector_opencv = np.array(airsim_translation_vector)

                # Stack R and t together to form the transformation matrix
                R = np.array([[0,1,0],[0,0,-1],[1,0,0]])
                transformation_matrix_opencv = np.eye(4)
                transformation_matrix_opencv[:3, :3] = rotation_matrix_opencv
                transformation_matrix_opencv[:3, 3] = translation_vector_opencv

                transformation_matrix_opencv[:3,:] = np.matmul(R,transformation_matrix_opencv[:3,:])
                poses.append(transformation_matrix_opencv)
            i += 1
            
        #print(poses)
    return poses
