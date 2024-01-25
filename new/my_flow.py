import cv2
import numpy as np
from new.PA2_1 import LK_OpticalFlow
import time
import os
import time
import progressbar
import matplotlib.pyplot as plt
#from scipy.optimize import least_squares
#from my_triangulation import triangulation
#from optimization import optimization
#from optimization import optimize
#from new import plot_opt
def graph(x,y,z):

    plt.subplot(1,3,1)
    plt.plot(y,x,'r')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1,3,2)
    plt.plot(z,x,'b')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.subplot(1,3,3)
    plt.plot(y,z,'g')
    plt.xlabel('y')
    plt.ylabel('z')
    
    # Show the plot
    plt.show()

def resize_array(array1,sz_array, num):
    array2 = np.zeros((sz_array.shape[0], sz_array.shape[1],sz_array.shape[2]), dtype=int)
    #print(sz_array.shape[0])
    # Calculate the coordinates for placing the 2x2 array at the center of the 4x4 array
    row_start = int((array2.shape[1] - array1.shape[1]) / num)
    row_end = row_start + array1.shape[1]
    col_start = int((array2.shape[2] - array1.shape[2]) / num)
    col_end = col_start + array1.shape[2]

    # Place the 2x2 array at the center of the 4x4 array
    array2[:,row_start:row_end, col_start:col_end] = array1
    return array2


def build_image_pyramid(image, num_levels):
    pyramid = [image]
    for _ in range(num_levels - 1):
        image = cv2.pyrDown(image)  # Downsample the image
        pyramid.append(image)
    return pyramid

def upscale_motion_vectors(motion_vectors, scaling_factor=2):
    return motion_vectors / scaling_factor

def pyramidal_lucas_kanade(image1, image2, num_levels, window_size):
    pyramid1 = build_image_pyramid(image1, num_levels)
    pyramid2 = build_image_pyramid(image2, num_levels)
    i = 0
    motion_vectors = np.zeros((2,image1.shape[0], image1.shape[1]),dtype=float)
    
    for level in range(num_levels - 1, -1, -1):
        scaled_image1 = pyramid1[level]
        scaled_image2 = pyramid2[level]
        
        # Estimate motion vectors at the current level
        #motion_vectors_low_res = LK_OpticalFlow(scaled_image1, scaled_image2)
        u,v,feature,dept, p1 = LK_OpticalFlow(scaled_image1, scaled_image2, window_size)
        #u = np.nan_to_num(u, nan=0)
        #v = np.nan_to_num(v, nan=0)
        #print("u = ", np.mean(u), " v = ", np.mean(v))
        motion_vectors_low_res = np.stack((u,v), axis=0)
        #print(np.shape(motion_vectors_low_res))
        # Scale the motion vectors for the next level
        #if level > 0:
        #    motion_vectors_low_res = upscale_motion_vectors(motion_vectors_low_res)
        
        # Add the current level's motion to the total motion field
        '''
        if motion_vectors.shape[1] != motion_vectors_low_res.shape[1]:
            number = motion_vectors.shape[1]/motion_vectors_low_res.shape[1]
            motion_vectors_low_res = resize_array(motion_vectors_low_res,motion_vectors,number)
        '''
        while motion_vectors.shape[1] != motion_vectors_low_res.shape[1]:
            motion_vectors_low_res = np.stack((cv2.pyrUp(motion_vectors_low_res[0,:,:]),cv2.pyrUp(motion_vectors_low_res[1,:,:])),axis=0)
            #print(np.mean(motion_vectors_low_res[0,:,:]))
            #motion_vectors_low_res[1,:,:] = cv2.pyrUp(motion_vectors_low_res[1,:,:])
        #'''
        #print(np.mean(motion_vectors_low_res[0,:,:]),np.mean(motion_vectors_low_res[1,:,:]))
        motion_vectors += motion_vectors_low_res
    return motion_vectors,feature,dept, p1

def plot(image,feature,dpet):

    # Define the array of arrows, each arrow as [x1, y1, dx, dy]
    s = np.shape(image)
    #arrows = []
    '''
    for i in range(s[0]):
        for j in range(s[1]):
            #if (u[i,j] != 0 or v[i,j] != 0 ) and (u[i,j] != np.nan or v[i,j] != np.nan) and (u[i,j] != np.inf or v[i,j] != np.inf):
            if ~(np.isnan(u[i,j]) or u[i,j] ==0) :
                #print(u[i,j],v[i,j])
                #arrows.append([i,j,int(v[i,j]),int(u[i,j])])
                cv2.arrowedLine(image, (j, i), (int(j+v[i,j]), int(i+u[i,j])), (0, 0, 255), 2, tipLength=0.2)
    '''
    dept = np.int0(np.array(dpet))
    feature = np.int0(np.array(feature))
    #print(np.shape(feature), np.shape(dept))
    #print((feature[0,0]), (feature[0,1]))
    for i in range(len(dept)):
        #print(np.shape(l))
        #j,i = l.ravel()
        #print( (int(feature[i,0]), int(feature[i,1])), (int(round(dept[i,0])), int(round(dept[i,1]))))
        cv2.arrowedLine(image, ((feature[i,0]),(feature[i,1])), ((dept[i,0]), (dept[i,1])), [255, 150, 0], 1, tipLength=0.2)
    '''
    # Loop through the arrows and draw them on the image
    for arrow in arrows:
        #print(arrow)
        x1, y1, dx, dy = arrow
        x2, y2 = x1 + dx, y1 + dy
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.2)  
        # (0, 0, 255) is the color (in BGR), 2 is the thickness, and tipLength is the length of the arrowhead
    '''
        # Display the image with arrows
    cv2.imshow('Optical flow Frame 2',image)
    #print("image show")
    key = cv2.waitKey(10)
    return key

def main():
    #t=0.3
    # Create a progress bar widget
    progress_bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Percentage(), '] ',
        progressbar.Bar(), ' (', progressbar.Timer() , ') ',
    ])
    image_directory = 'KITTI_sequence_1/image_l'
    #'images'
    #'image_2'
    #'KITTI_sequence_2/image_l'
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))])
    # Loop through image files and add arrows to each image
    cam_pose = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float32)
    ca_pose = np.array([ 0.00, 0.00, 0.00], dtype=np.float32)
    initial_pose = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
    pose_x = [0.0]
    pose_y = [0.0]
    pose_z = [0.0]
    for i in progress_bar(range(len(image_files)-1)):
        image_fil = image_files[i]
        next_imag = image_files[i+1]
        image_file = cv2.imread(image_directory + '/' + image_fil)
        next_image = cv2.imread(image_directory + '/' + next_imag)
        Imag1 = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        Imag2 = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        Image1 = np.array(Imag1)
        Image2 = np.array(Imag2)
        motion,feature,dept = pyramidal_lucas_kanade(Image1,Image2,1,3)
        u = motion[0,:,:]
        v = motion[1,:,:]
        fx=fy=150
        cx=150
        cy=100
        #image_path = os.path.join(image_directory,'/', image_file)
        #frame = cv2.imread(image_path)
        #print("One")
        cam_pose += triangulation(feature,dept)
        #print(cam_pose)
        #print("Two")
        #cam_pos = optimization(initial_pose,world_coordinates,feature,fx,fy,cx,cy)
        #print(np.shape(world_coordinates))
        #print(np.shape(two_d_points))

        #if ((abs(cam_pos[3]) < 2) and (abs(cam_pos[4]) < 2) and (abs(cam_pos[5]) < 2)):
        #print('\n',cam_pos)
        print('\n',cam_pose)
        #cam_pose += 0.001 * cam_pos
        #print("Three")
        #ca_pose += cam_pos
        pose_x.append(cam_pose[3])
        pose_y.append(cam_pose[4])
        pose_z.append(cam_pose[5])
        #print("four")
        key = plot(image_file,feature,dept)
        #print('\n',cam_pos)
        
        '''
        points_3d,_ = triangulation(feature, dept)
        point_3d = np.zeros((len(points_3d[0,0]),3))
        for j in range(len(points_3d)):
            point_3d[j] = (points_3d[0,0,j],points_3d[1,0,j],points_3d[2,0,j])
        '''
        # Check for key press events
        #key = cv2.waitKey(10)
        
        if key == 27:
            break
        elif key == 32:  # Press 'Space' to pause/unpause (toggle)
            while True:
                key2 = cv2.waitKey(0)
                if key2 == 32 or key2 == 27:
                    break
    
        time.sleep(0.1)
        #print("hey")
        # Increment the frame index
        #frame_index += 1
    progress_bar.finish()
    res_list = [sum(i) for i in zip(pose_y , pose_z)]
    #print((pose_x),pose_y)
    graph(np.array(pose_x),np.array(pose_y),np.array(pose_z))
    #print('\n',np.array(pose_x),'\n',np.array(pose_y),'\n',np.array(pose_z))
    #plot_opt(np.array(pose_x),np.array(pose_y))
    #plt.plot(pose_x, res_list)
    plt.show()

    # Release the window and close it
    #cv2.destroyWindow('Image with Arrow')






if __name__ == "__main__":
    main()


'''def triangulation(feature, dept):
    point1 = []
    point2 = []
    for i in range(len(feature)):
        point1.append(feature[i])
        point2.append(point1[i] + dept[i])

    points1 = np.array(point1)
    points2 = np.array(point2)

    # Calculate the essential matrix (E) or the fundamental matrix (F)
    E, mask = cv2.findEssentialMat(points1, points2, focal=1.0, pp=(0, 0))

    # Recover relative pose (rotation and translation) between cameras
    _, R, t, _ = cv2.recoverPose(E, points1, points2)

    # Create camera projection matrices
    #P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # Create the camera intrinsic matrix K
    fx=fy=120
    cx=120
    cy=80
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    # Define the camera's rotation matrix and translation vector
    # For simplicity, we'll use an identity rotation matrix and a translation of (0, 0, 5) units
    #R = np.eye(3)  # Identity rotation matrix
    #t = np.array([0, 0, 5])  # Translation vector

    # Create the camera projection matrix P1
    P1 = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
    print(P1)
    #print(R)
    #print(t.reshape(-1, 1))
    P2 = np.hstack((R, t.reshape(-1, 1)))
    cw = -R.T*t
    
    # Projection matrices for the two cameras
    P1 = K.dot(np.hstack((np.identity(3), np.zeros((3, 1)))))
    P2 = K.dot(np.hstack((R, t.reshape(-1,1))))
    # Triangulate points to find 3D coordinates
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

    points_3d = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T).reshape(-1, 3)
    # Convert homogeneous coordinates to 3D Cartesian coordinates
    #points_3d = (points_3d_homogeneous[:3, :] / (points_3d_homogeneous[3, :] )).reshape(-1,3)
    # Now, points_3d contains the 3D coordinates of the corresponding points
    #print(points_3d)
    return points_3d
    

def optimization(initial_pose,feature,world_coordinates,image_coordinates,fx,fy,cx,cy):

    def euler_to_rotation_matrix(roll, pitch, yaw):
        # Convert Euler angles to rotation matrix (ZYX order)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Define the rotation matrix
        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr]])
        
        return R


    # Define the objective function to minimize (sum of squared errors)
    # Define the objective function to minimize (sum of squared errors)
    # Define the objective function to minimize (sum of squared errors)
    def objective_function(params, world_coords, image_coords, fx, fy, cx, cy):
        # Extract pose parameters (rotation angles and translation)
        roll, pitch, yaw, tx, ty, tz = params

        # Create the rotation matrix from Euler angles
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        #if image_coords.all() == feature.all():
        #    print("Same")

        # Transform world coordinates to camera coordinates
        camera_coords = np.dot(R, world_coords.T - np.array([tx, ty, tz]).reshape(3, 1))
        
        # Project camera coordinates onto the image plane
        u = (fx * camera_coords[0, :] / camera_coords[2,:]) + cx
        v = (fy * camera_coords[1, :] / camera_coords[2,:]) + cy

        # Stack u and v to form the (N, 2) array of projected image coordinates

        projected_coords = np.zeros((len(u),1,2))
        (projected_coords[:,0,0]) = u
        (projected_coords[:,0,1]) = v
        #projected_coords = np.nan_to_num(projected_coords, nan=0)
        #print(projected_coords)

        # Calculate the error (squared differences) between observed and computed 2D projections
        error = np.ravel(image_coords - projected_coords)

        return error

    # Perform least-squares optimization
    result = least_squares(objective_function, initial_pose, args=(world_coordinates, image_coordinates, fx, fy, cx, cy))

    # Extract the optimized pose parameters
    optimized_pose = result.x
    #print(optimized_pose)
    # The optimized_pose now represents the camera's pose (position and orientation) that best fits the observed feature positions
    return optimized_pose
    
    
def optimization(world_coordinates,image_coordinates):
    # Camera intrinsic parameters (focal length, principal point)
    fx = 120
    fy = 120
    cx = 120
    cy = 80

    # Initial guess for camera pose (rotation and translation)
    initial_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    # Define the objective function to minimize (sum of squared errors)
    def objective_function(params):
        # Extract pose parameters (rotation angles and translation)
        roll, pitch, yaw, tx, ty, tz = params
        
        # Calculate the 2D projections from 3D world coordinates
        projected_coordinates = project_world_to_image(world_coordinates, fx, fy, cx, cy, roll, pitch, yaw, tx, ty, tz)
        
        # Calculate the error (squared differences) between observed and computed 2D projections
        error = np.ravel(image_coordinates - projected_coordinates)

        return error

    # Perform least-squares optimization
    result = least_squares(objective_function, initial_pose)

    # Extract the optimized pose parameters
    optimized_pose = result.x
'''

    
