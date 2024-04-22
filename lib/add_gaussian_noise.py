import numpy as np
from scipy.spatial.transform import Rotation as R




def pose_decomposition(hom_matrix):
    rmat = hom_matrix[:3, :3]
    tvec = hom_matrix[:3, 3]

    return rmat, tvec


def rod_to_rmat(rod_vec):
    # Initialize a Rotation object with the Rodrigues vector
    rotation = R.from_rotvec(rod_vec)
    
    # Convert to rotation matrix representation
    rmat = rotation.as_matrix()
    
    # Return the rotation matrix
    return rmat


def rmat_to_rod(rmat):
    """
    Converts a rotation matrix to a Rodrigues vector.
    
    Parameters:
    - rmat: The rotation matrix.
    
    Returns:
    - Rodrigues vector representing the rotation.
    """
    # Initialize a Rotation object with the rotation matrix
    rotation = R.from_matrix(rmat)
    
    # Convert to Rodrigues vector representation
    rod_vec = rotation.as_rotvec()
    
    # Return the Rodrigues vector
    return rod_vec


def add_gaussian_noise_to_tvec(tvec, std_dev):
    """
    This function adds Gaussian distributed translation noise with zero mean to 
    the translation vectors.

    Parameters:
    - tvec: The original translation vector.
    - std_dev: The standard deviation of the Gaussian noise.

    Returns:
    - Noisy translation vector with Gaussian noise added.
    """

    # Generate Gaussian distributed random noise for the translation vector
    # np.random.normal takes mean, standard deviation and output shape as arguments
    translation_noise = np.random.normal(0, std_dev, tvec.shape)

    # Add noise to the translation vector
    noisy_translation_vector = tvec + translation_noise

    return noisy_translation_vector


def add_gaussian_noise_to_rot_vector(rot_vec, std_dev):
    """
    This function adds Gaussian distributed rotation noise with zero mean to 
    the rotation axes.

    Parameters:
    - rot_axis: The original rotation axes vector (Euler angles).
    - std_dev: The standard deviation of the Gaussian noise.

    Returns:
    - Noisy rotation axes vector with Gaussian noise added.
    """

    # Generate Gaussian distributed random noise for the rotation axes
    # np.random.normal takes mean, standard deviation and output shape as arguments
    rotation_noise = np.random.normal(0, std_dev, rot_vec.shape)

    # Add noise to the rotation axes
    noisy_rot_vec = rot_vec + rotation_noise

    return noisy_rot_vec



###############################################################################
######       Decompose Pose and Add Noise  to tvec and rot_vec         ########
###############################################################################


# 0.001rad is about 0.0573 deg

# Standard deviation for Gaussian noise

# We will vary the noise in the translation and rotation pose components
# Rotation:    (0, pi/180, 2pi/180, 3pi/180, 4pi/180 and 5pi/180)rad std dev
# Translation: (0, 1, 2, 3, 4, 5)mm std dev 

# Rotation noise standard deviation(Adjust the deviations as desired)
dev_in_degree = 1           # Eg 1deg
std_dev_rvec = dev_in_degree * np.pi/180

 # Translation noise standard deviation(Adjust the deviations as desired)
std_dev_tvec = 1            # Eg 1mm


     

    


# Decompose the 4x4 homogenous poses into 3x3 rmat and 3x1 tvec for Camera
rmat_camera = []
tvec_camera = []
def gaussian_noise(cameraposes)

    for item in camera_poses:
        rmat_camera_temp, tvec_camera_temp = pose_decomposition(item)
        
        # Convert rot_mat to vector
        rvec_camera_temp = rmat_to_rod(rmat_camera_temp)
        #rvec_camera_temp = rvec_tcp_temp.reshape((3,1))
        
        # Add normally distributed noise to the rotation axis
        rvec_camera = add_gaussian_noise_to_rot_vector(rvec_camera_temp, std_dev_rvec)
        
        # Convert rot_vector back to matrix
        rmat_camera_temp = rod_to_rmat(rvec_camera)
        rmat_camera.append(rmat_camera_temp)
        
        
        
        # Add normally distributed noise to the translation vector
        tvec_camera_temp = tvec_camera_temp.reshape(3,1)
        tvec_camera_temp = add_gaussian_noise_to_tvec(tvec_camera_temp, std_dev_tvec)
        tvec_camera.append(tvec_camera_temp)

