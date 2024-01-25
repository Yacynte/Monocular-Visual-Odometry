"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wueryburg-Schweinfurt ---------"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from csv_poses import get_opt_gt, get_airsim_gt

from scipy.spatial.transform import Rotation
from lib.pyr_lucas_kanade import lucas_pyramidal
from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,'path_gts/semi_circle_08_01_002.csv')
                                                      ,os.path.join(data_dir,'timestamps/semi_circle_08_01_002.csv'))
        self.images = self._load_images(os.path.join(data_dir,"images/semi_circle_08_01_002"))
        
    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath,timestamp):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        #print(len(filepath))
        #time_stamp, v = 0, 0
        #timestamp = "KITTI_sequence_1/timestamps/semi_circle_02_01.csv"
        
        if filepath[len(filepath)-4:] == ".csv":
            poses = get_opt_gt(filepath,timestamp)
            #poses = poses.tolist()
        elif filepath[len(filepath)-7:] == "rec.txt":
            poses = get_airsim_gt(filepath)
        else:
            poses = []
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    T = np.fromstring(line, dtype=np.float64, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    #print(type(T))
                    poses.append(T)
        
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path) for path in image_paths]
        #return  [cv2.resize(img, (640,480), interpolation= cv2.INTER_LINEAR) for img in image]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i, step, initial_guess):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the Lucas Kanade optical flow function

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints position in i-1'th image
        q2 (ndarray): The good keypoints position in i'th image
        """

        q1 , q2, foes, n, next_guess = lucas_pyramidal(self.images[i-step], self.images[i+1-step],5 ,5 ,70, 3, initial_guess)

        """This function plots the optical flow and on the i'th image"""
        key = plot(self.images[i-step], np.copy(q1), np.copy(q2), foes, n, i)
        #print(time1-time0)
        #key = 0
        return q1, q2, key, next_guess

    def get_pose(self, q1, q2, z_c=0.028):
        # z_c is the distance between the center of the camera and the center of the optitrack markers. This is an approximate that was measured using a ruler
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2, z_c)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2, z_c):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(np.float32(self.P), np.float32(P), np.float32(q1.T), np.float32(q2.T))
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)
            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     (np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        t = t*z_c
        
        # Make a list of the different possible pairs
        pairs = [[R1, -t], [R1, t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        #print(right_pair_idx)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return R1, t

def plot(image,q3,q4,foes,n, j):
    s = np.shape(image)
    feature = np.intp(q3)
    dept = np.intp(q4)

    # Draw arrow on image
    for i in range(len(dept)):
        cv2.arrowedLine(image, ((feature[i,0]),(feature[i,1])), ((dept[i,0]), (dept[i,1])), [255, 150, 0], 1, tipLength=0.2)
    
    # Draw focus of expansion
    cv2.circle(image, (np.intp(foes[0]),np.intp(foes[1])), 7, (0, 0, 255), -1)
    # Display the image with arrows
    cv2.imshow('Optical flow Frame',image)
    #print("image show")
    key = cv2.waitKey(10)
    return key


def main():
    data_dir = "Sequence_paths"  # Try KITTI dataset, KITTI_sequence
    vo = VisualOdometry(data_dir)
    
    poses = vo.gt_poses
    poses_ = np.copy(poses) 
    startframe = 100
    step = 1
    endframe = 20 # stop thes number of frames befor the last frame
    poses = np.array(poses)
    poses = poses[startframe:-(endframe+1):step,:,:]

    initial_guess = np.zeros((10,2), dtype=np.float32)
    gt_path_3d = []
    estimated_path_3d = []

    for i, gt_pose in zip(range(startframe,len(poses_) - endframe,step),(tqdm(poses, unit=" pose"))):
    
        if i == startframe:
            cur_pose = gt_pose
            
        else:
            q1, q2, key, initial_guess = vo.get_matches(i,step, initial_guess)
            transf = vo.get_pose(q1, q2 )
            cur_pose = np.matmul(np.linalg.inv(transf),cur_pose)
            if key == 27:
                break
            elif key == 32:  # Press 'Space' to pause/unpause (toggle)
                while True:
                    key2 = cv2.waitKey(0)
                    if key2 == 32 or key2 == 27:
                        break

        gt_path_3d.append((gt_pose[0, 3], gt_pose[1,3], gt_pose[2, 3]))
        estimated_path_3d.append((cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))


    """----------Rotate the estimates pose to aligne with the ground truth. This will be replaced by Hand Eye Callibration----------"""    
    # Create a rotation object for Z-axis rotation
    rotation_z = Rotation.from_euler('z', np.radians(0))

    # Create a rotation object for X-axis rotation
    rotation_x = Rotation.from_euler('x', np.radians(-5))

    # Create a rotation object for X-axis rotation
    rotation_y = Rotation.from_euler('y', np.radians(0))

    # Combine the two rotations (Z followed by X)
    combined_rotation = rotation_x * rotation_y * rotation_z

    # Get the resulting rotation matrix
    R = combined_rotation.as_matrix()
    
    # Apply the Rotation
    gt_path_3d = np.array(gt_path_3d)
    estimated_path_3d = np.array(estimated_path_3d) - gt_path_3d[0]
    estimated_path_3d = np.matmul(R,estimated_path_3d.T)
    estimated_path_3d = estimated_path_3d.T  + gt_path_3d[0]

    """-----------Plot the Results to view the estimated path alongside the ground truth----------"""
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(estimated_path_3d[:,2],estimated_path_3d[:,0],estimated_path_3d[:,1],label='Visual Odometry')
    ax.plot(gt_path_3d[:,2],gt_path_3d[:,0],gt_path_3d[:,1],label='Ground Truth')
    #plt.grid()
    ax.legend()
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    plt.show()
    

if __name__ == "__main__":
    main()
