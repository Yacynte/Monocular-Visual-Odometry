"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""
"""---------- Centre for Robotics ---------"""

from cProfile import label
import os
import time
import numpy as np
import queue
import cv2
import queue
import matplotlib.pyplot as plt
from csv_poses import get_opt_gt, get_airsim_gt

from scipy.spatial.transform import Rotation
from lib.pyr_lucas_kanade import lucas_pyramidal
# from lib.add_gaussian_noise import a
from tqdm import tqdm

 
class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, "calib.txt"))
        self.gt_poses, self.imu_distance = self._load_poses([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.startswith("ground_truth")]
                                                      ,[os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.startswith("imu")])
        self.images = self._load_images(os.path.join(data_dir,"images"))
        
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
    def _load_poses(filepath_,timestamp_):
        filepath = filepath_[0]
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """        
        if "Tello_dataset" in filepath:
            poses, dist = get_opt_gt(filepath,timestamp_[0])
        elif "UE_dataset" in filepath:
            poses, dist = get_airsim_gt(filepath)
        else:
            poses = []
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    T = np.fromstring(line, dtype=np.float64, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    #print(type(T))
                    poses.append(T)
            gt_path_3d = np.array(poses)
            dist = []
            for i in range(1,len(gt_path_3d)):
                dist.append(np.linalg.norm(gt_path_3d[i,:3,3]-gt_path_3d[i-1,:3,3]))
        # print(poses)
        return poses, dist

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

    def get_matches(self, i, step, number_features=200, wz=5, level=5, number_iteration=70, inlier_threshold=3, static=False):
        """
        This function detect and compute keypoints from the i-1'th and i'th image using the Lucas Kanade optical flow function

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints position in i-1'th image
        q2 (ndarray): The good keypoints position in i'th image
        """

        q1 , q2, foes = lucas_pyramidal(self.images[i-step], self.images[i+1-step], number_features, wz ,level ,number_iteration, inlier_threshold, static)

        """This function plots the optical flow and on the i'th image"""
        key = plot(self.images[i-step], np.copy(q1), np.copy(q2), foes, static)
        #print(time1-time0)
        #key = 0
        return q1, q2, key

    def get_pose(self, q1, q2):
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
        E, _ = cv2.findEssentialMat(q1, q2, self.K, method=0, threshold=0.1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        # transformation_matrix = self._form_transf(R, np.squeeze(t))
        return R, t

    def decomp_essential_mat(self, E, q1, q2):
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

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        #t = t*z_c
        
        # Make a list of the different possible pairs
        pairs = [[R1, -t], [R1, t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        #print(right_pair_idx)
        right_pair = pairs[right_pair_idx]
        R1, t = right_pair

        return R1, t

def plot(image,q3,q4,foes, static):
    S = np.shape(image)
    s = np.intp(np.intp(S)/3)
    feature = np.intp(q3)
    dept = np.intp(q4)

    # Draw arrow on image
    for i in range(len(dept)):
        cv2.arrowedLine(image, ((feature[i,0]),(feature[i,1])), ((dept[i,0]), (dept[i,1])), [255, 150, 0], 1, tipLength=0.2)
    
    # Draw focus of expansion
    if static:
        cv2.circle(image, (np.intp(foes[0]),np.intp(foes[1])), 7, (0, 0, 255), -1)
    else:
        # Draw the line on the image
        image_with_line = cv2.line(image, (0,s[0]), (S[1], s[0]), (0, 0, 255), thickness=2)
        image_with_line = cv2.line(image, (0,2*s[0]), (S[1], 2*s[0]), (0, 0, 255), thickness=2)
        image_with_line = cv2.line(image, (s[1],0), (s[1], S[0]), (0, 0, 255), thickness=2)
        image_with_line = cv2.line(image, (2*s[1],0), (2*s[1], S[0]), (0, 0, 255), thickness=2)
        for j in range(len(foes)):
            cv2.circle(image, (np.intp(foes[j,0]),np.intp(foes[j,1])), 7, (0, 0, 255), -1)
    # Display the image with arrows
    cv2.imshow('Optical flow Frame',image)
    #print("image show")
    key = cv2.waitKey(10)
    return key


def main(queued_data, data_dir, gt_path_3d, estimated_path_3d, number_features=200, wz=5, level=5, iteration=70, inlier_threshold=3, static = False):
    #data_dir = "UE_square_path"  # Try KITTI dataset, KITTI_sequence , UE dataset UE_square_path, Tello dataset Sequence_paths
    vo = VisualOdometry(data_dir)
    imu_dist = vo.imu_distance
    # imu_speed = vo.imu_velocity
    poses = vo.gt_poses
    poses_ = np.copy(poses) 
    startframe = 5
    step = 1
    endframe = 1 # stop at this number of frames befor the last frame
    if "Tello" in data_dir:
        endframe = 20
    poses = np.array(poses)
    poses = poses[startframe:-(endframe):step,:,:]
    
    local_t = []
    path_imu = []
    gt_path_3d_up = []
    # t0 = time.time()
    t1 = 1
    estimated_rot = []
    gt_rot = []

    for i, gt_pose in zip(range(startframe,len(poses_) - endframe,step),(tqdm(poses, unit=" Frames", smoothing=0))):

        shared_data = [i-startframe, len(poses), 1/t1]
        queued_data.put(shared_data)

        if i == startframe:
            j = startframe
            cur_pose = gt_pose
            
        else:
            t0 = time.time()
            # get correspondin points
            q1, q2, key = vo.get_matches(i,step, number_features, wz, level, iteration, inlier_threshold, static)
            # estimate the motion
            R, t = vo.get_pose(q1, q2)
            # scale of motion        
            t = t*imu_dist[i]
            # motion reconstruction
            transf = vo._form_transf(R, np.squeeze(t))
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            t1 = time.time() - t0
            if key == 27:
                break
            elif key == 32:  # Press 'Space' to pause/unpause (toggle)
                while True:
                    key2 = cv2.waitKey(0)
                    if key2 == 32 or key2 == 27:
                        break

        # gt_rot_ = Rotation.from_matrix(gt_pose[:3,:3]).as_rotvec()
        # estimated_rot_ = Rotation.from_matrix(cur_pose[:3,:3]).as_rotvec()
        # gt_rot_ = Rotation.from_matrix(gt_pose[:3,:3]).as_rotvec()
        gt_rot_, _ = cv2.Rodrigues(gt_pose[:3,:3])
        estimated_rot_, _ = cv2.Rodrigues(cur_pose[:3,:3])
        gt_rot.append(gt_rot_)
        estimated_rot.append(estimated_rot_)
        # estimated_path_3d.append(cur_pose)
        # gt_path_3d.append(gt_pose)
        estimated_path_3d.append((estimated_rot_[0][0], estimated_rot_[1][0], estimated_rot_[2][0], cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))
        gt_path_3d.append((gt_rot_[0][0], gt_rot_[1][0], gt_rot_[2][0], gt_pose[0, 3], gt_pose[1,3], gt_pose[2, 3]))
        # estimated_path_3d.append((estimated_rot_[0], estimated_rot_[1], estimated_rot_[2]))
        # gt_path_3d.append((gt_rot_[0], gt_rot_[1], gt_rot_[2]))
    

if __name__ == "__main__":
    main()
