"""---------- Batchaya Noumeme Yacynte Divan ----------"""
"""---------- Bachelor's in Mechatronics Thesis ----------"""
"""---------- Development of Monocular Visual Odometry Algorithm, WS 23/24 ----------"""
"""---------- Technische Hochshule Wuerzburg-Schweinfurt ---------"""
"""---------- Centre for Robotics ---------"""

from cProfile import label
import tkinter as tk
import cv2, csv
from scipy.spatial.transform import Rotation
import time
from tkinter import ttk
from tkinter import filedialog
from functools import partial
import numpy as np
import os
import queue
from tkinter.ttk import Progressbar
import threading
import matplotlib.pyplot as plt
import visual_odometry  
from plot_param import regression

class VOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Odometry")
       # Selected data folder path
        self.data_folder_path = tk.StringVar()

        # Create frame for the option selection on the left
        self.option_frame_left = ttk.Frame(root)
        self.option_frame_left.pack(side="left", padx=20, pady=20)

        # Label for the data folder selection
        #ttk.Label(self.option_frame_left, text="Select Data Folder:").grid(row=0, column=0, padx=5, pady=5)
        # Progress bar
        self.progress_label = ttk.Label(self.option_frame_left, text="Progress Bar:")
        self.progress_label.grid(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = Progressbar(self.option_frame_left, length=200, mode='determinate', variable=self.progress_var)
        self.progress_bar.grid(pady=5, columnspan=3)

        # Create a label to display the percentage
        self.percentage_label = tk.Label(self.option_frame_left, text="")
        self.percentage_label.grid(row=2, column=0, padx=5, pady=5)

        # Create a label to display the speed   
        self.speed_label = tk.Label(self.option_frame_left, text="")
        self.speed_label.grid(row=2, column=1, padx=5, pady=5)

        # Entry for the selected data folder path
        # self.data_folder_entry = ttk.Entry(self.option_frame_left, textvariable=self.data_folder_path, state="readonly")
        # self.data_folder_entry.grid(row=0, column=1, padx=5, pady=5)

        # Button to browse and select data folder
        # ttk.Button(self.option_frame_left, text="Browse", command=self.select_data_folder).grid(row=0, column=2, padx=5, pady=5)

        # Button to clear selected data folder
        # ttk.Button(self.option_frame_left, text="Clear Data Folder", command=self.clear_data_folder).grid(row=0, column=3, padx=5, pady=5)

        # Label for the option selection
        ttk.Label(self.option_frame_left, text="Tello Dji data:").grid(row=3, column=0, padx=5, pady=5)
        # Combobox for Option 1
        self.option1_var = tk.StringVar()
        self.option1_combo = ttk.Combobox(self.option_frame_left, textvariable=self.option1_var, state="readonly")
        self.option1_combo.grid(row=3, column=1, padx=5, pady=5)
        self.option1_combo['values'] = ('c-curve','line')#, 'line2', 'square', 'c_curve_3')

        # Label for the sub-option selection
        ttk.Label(self.option_frame_left, text="Kitti data:").grid(row=4, column=0, padx=5, pady=5)

        # Combobox for Option 2
        self.option2_var = tk.StringVar()
        self.option2_combo = ttk.Combobox(self.option_frame_left, textvariable=self.option2_var, state="readonly")
        self.option2_combo.grid(row=4, column=1, padx=5, pady=5)
        self.option2_combo['values'] = ('c-curve', 'line')

        # Label for the sub-option selection
        ttk.Label(self.option_frame_left, text="Unreal Engine data:").grid(row=5, column=0, padx=5, pady=5)

        # Combobox for Option 3
        self.option3_var = tk.StringVar()
        self.option3_combo = ttk.Combobox(self.option_frame_left, textvariable=self.option3_var, state="readonly")
        self.option3_combo.grid(row=5, column=1, padx=5, pady=5)
        self.option3_combo['values'] = ('line', 'square')#, 'c-curve', 'circle', 'square_rotate')

        # Button to select the folder based on the options
        #ttk.Button(self.option_frame_left, text="Select dataset", command=self.select_folder).grid(pady=10)
        # Button to unselect the option
        ttk.Button(self.option_frame_left, text="Unselect data", command=self.clear_option_selection).grid(pady=10)

        self.switch_state = True
        self.toggle_button = tk.Button(self.option_frame_left, text="STATIC", command=self.toggle_switch)
        self.toggle_button.grid(row=6, column=1, pady=10)
        self.switch_label = tk.Label(self.option_frame_left, text="STATIC", bg="red", width=10)
        self.switch_label.grid(row=6, column=2, pady=10)


        self.update_switch_label()


        # Create frame for the option selection on the right
        self.option_frame_right = ttk.Frame(root)
        self.option_frame_right.pack(side="right", padx=20, pady=20)

        """# Button to unselect the option
        ttk.Button(self.option_frame_right, text="Unselect Option", command=self.clear_option_selection).grid(row=0, column=0, padx=5, pady=5)

        # Label for the option selection
        ttk.Label(self.option_frame_right, text="Select Option 1:").grid(row=1, column=0, padx=5, pady=5)

        # Combobox for Option 1
        self.option1_var = tk.StringVar()
        self.option1_combo = ttk.Combobox(self.option_frame_right, textvariable=self.option1_var, state="readonly")
        self.option1_combo.grid(row=1, column=1, padx=5, pady=5)
        self.option1_combo['values'] = ('Option 1.1', 'Option 1.2', 'Option 1.3')

        # Button to select the folder based on the options
        ttk.Button(root, text="Select Folder", command=self.select_folder).pack(pady=10)
"""

        self.start_button = ttk.Button(self.option_frame_right, text="Start VO", command=self.start_vo)
        self.start_button.pack(pady=5)

        self.pause_button = tk.Button(self.option_frame_right, text="Pause VO", command=self.pause_vo, state=tk.DISABLED)
        self.pause_button.pack(pady=5)

        self.stop_button = tk.Button(self.option_frame_right, text="Stop VO", command=self.stop_vo, state=tk.DISABLED)
        self.stop_button.pack(pady=5)


        # Label for the option selection
        # ttk.Label(self.option_frame, text="Select Option 1:").grid(row=0, column=0, padx=5, pady=5)

        
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)
    

        # Input parameters
        self.param_label = tk.Label(root, text="Input Parameters:")
        self.param_label.pack(pady=5)

        # Initial values for parameters
        initial_param1 = 500 # number of features
        initial_param2 = 5 # window size
        initial_param3 = 5 # level
        initial_param4 = 70 # iteration
        initial_param5 = 3 # inlier threshold


        self.param1_label = tk.Label(root, text="Number of Features:")
        self.param1_label.pack()
        self.param1_entry = tk.Entry(root)
        self.param1_entry.insert(0, initial_param1)  # Set initial value
        self.param1_entry.pack()

        self.param2_label = tk.Label(root, text="Optical Flow window size:")
        self.param2_label.pack()
        self.param2_entry = tk.Entry(root)
        self.param2_entry.insert(0, initial_param2)  # Set initial value
        self.param2_entry.pack()

        self.param3_label = tk.Label(root, text="level of Pyramid:")
        self.param3_label.pack()
        self.param3_entry = tk.Entry(root)
        self.param3_entry.insert(0, initial_param3)  # Set initial value
        self.param3_entry.pack()

        self.param4_label = tk.Label(root, text="Number of Iteration:")
        self.param4_label.pack()
        self.param4_entry = tk.Entry(root)
        self.param4_entry.insert(0, initial_param4)  # Set initial value
        self.param4_entry.pack()

        self.param5_label = tk.Label(root, text="Inlier Threshold:")
        self.param5_label.pack()
        self.param5_entry = tk.Entry(root)
        self.param5_entry.insert(0, initial_param5)  # Set initial value
        self.param5_entry.pack()

        

        # Initialize thread variables
        self.vo_thread = None
        self.running = False
        self.pause = False
        self.stop = False
        self.time = time.time()
        self.folder_path = " "
        
        #self.q = queue.Queue()

        # Add more entry fields for additional parameters as needed
    def clear_option_selection(self):
        self.option1_var.set("")
        self.option2_var.set("")
        self.option3_var.set("")

    def update_progress_bar(self, progress_percent, shared_data):
        self.progress_bar['value'] = progress_percent
        self.root.update_idletasks()
        speed = shared_data[2]
        self.progress_var = progress_percent
        progress_percent = "{:.0f}".format(progress_percent)
        speed = "{:.2f}".format(speed)
        self.percentage_label.config(text=f"Progress: {progress_percent}%")
        self.speed_label.config(text=f"Speed: {speed} Frames/sec")

    def toggle_switch(self):
        # global switch_state
        self.switch_state = not self.switch_state
        self.update_switch_label()

    def update_switch_label(self):
        if self.switch_state:
            self.switch_label.config(text="ON", bg="green")
        else:
            self.switch_label.config(text="OFF", bg="red")


    def start_vo(self):
        if self.vo_thread is None or not self.vo_thread.is_alive():
            self.running = True
            self.stop = False
            self.pause = False
            self.time = time.time()
            # Create a queue
            q = queue.Queue()
            # print("Queue created")
            ground_truth = []
            monocular_vo = []
            param1 = self.param1_entry.get()
            param2 = self.param2_entry.get()
            param3 = self.param3_entry.get()
            param4 = self.param4_entry.get()
            param5 = self.param5_entry.get()
            # Get input parameters from entry fields

            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="VO Running...")


            # Get selected options
            option1 = self.option1_var.get()
            option2 = self.option2_var.get()
            option3 = self.option3_var.get()

            if option1:
                self.folder_path = "data/Tello_dataset/" + option1
            elif option2:
                self.folder_path = "data/Kitti_dataset/" + option2
            elif option3 :
                self.folder_path = "data/UE_dataset/" + option3
            
                

            #ue_data = "Monocular-Visual-Odometry/data/UE_virtual_data/" + option3
            #print(folder_path)
            #print(option1,option2,option3)
            # Construct the folder path based on selected options
            #folder_path = os.path.join("vo_package", option1, option2, option3)

            # Check if the folder exists
            if not os.path.exists(self.folder_path):
                tk.messagebox.showerror("Error", "Selected folder does not exist.")
                self.stop_vo()
                
            
            # Run VO in a separate thread with input parameters
            if self.running:
                
                self.vo_thread = threading.Thread(target=self.run_vo, args=(q, ground_truth, monocular_vo, param1, param2, param3, param4, param5,self.folder_path))
                self.vo_thread.daemon = True
                self.vo_thread.start()
                #print(self.running)
                #time.sleep(10)
                while self.running:
                    if self.pause_button.cget("state") != "normal":
                            self.pause_vo()
                    elif self.stop_button.cget("state") != "normal":
                        self.stop_vo()
                    else :
                        try:
                            # Update progress bar
                            shared_data = q.get(timeout=0.1)
                            progress_percent = (shared_data[0]+ 1) * 100 / shared_data[1]
                            # print(progress_percent, shared_data)
                            self.update_progress_bar(progress_percent, shared_data)
                            #print(self.pause_button.cget("state"))
                            state = [self.pause_button.cget("state"), self.stop_button.cget("state")]
                        except queue.Empty:
                            # Queue is empty
                            # print("No data available")
                            if not self.running or not self.vo_thread.is_alive() or self.stop_button.cget("state") != "normal" :
                                self.vo_finished()
                                break  # Exit loop if thread has finished

    def pause_vo(self):
        # Implement pause functionality here
        self.pause = True
        pass

    def stop_vo(self):
        #print("In stop")
        self.running = False
        self.stop = True
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.update_progress_bar(0)
        self.status_label.config(text="VO Stopped")
        root.mainloop()
        

    def run_vo(self, q, ground_truth, monocular_vo, param1, param2, param3, param4, param5, folder_path):
        # Your Visual Odometry logic goes here
        # Example: your_vo_module.run_vo(param1, param2)
        # print("Queue is here")
        visual_odometry.main(q, folder_path, ground_truth, monocular_vo, int(param1), int(param2), int(param3), int(param4), float(param5), self.switch_state,)
        # Close all openCV windows
        cv2.destroyAllWindows()
        # plt.close('all')
        """---------------- Hand Eye Callibration -------------"""
        
        gt_3d = np.array(ground_truth)
        gt_path_3d = gt_3d[:,3:]
        gt_rot_vec = gt_3d[:,:3]
        estimated_3d = np.array(monocular_vo)
        estimated_path_3d = estimated_3d [:,3:]
        estimated_rot_vec = estimated_3d[:,:3]
        vo_to_rotate = np.copy(estimated_path_3d - gt_path_3d[0])

        # est_rot_matrix, _ = cv2.Rodrigues(estimated_rot_vec)
        # gt_rot_matrix, _ = cv2.Rodrigues(gt_rot_vec)
        rotate_matrix = []
        for i in range(len(estimated_rot_vec)):
            R, _ = cv2.Rodrigues(estimated_rot_vec[i]-gt_rot_vec[0]) 
            rotate_matrix.append(R) # for i in range(len(estimated_rot_vec))]
        # print(estimated_rot_vec[i])
        rotate_matrix = np.array(rotate_matrix)
        hec_angles = np.array([np.radians(-3.7582), np.radians(12.5854), np.radians(4.9001)])
        hec_translation = np.array([-0.9762, 7.7199, -3.6327])/100
        if "UE" in self.folder_path:
            noise = np.random.normal(0,0.01,3*len(vo_to_rotate[21:]))
            noise = noise.reshape((len(vo_to_rotate)-21,3))
            vo_to_rotate[21:] += noise
            hec_rotation_matrix = Rotation.from_euler('xyz', [-90, 0, 90], degrees=True) #[-4, 12.5, 5], degrees=True)
            # hec_rotation_matrix = Rotation.from_euler('xyz', [90, 0, -90], degrees=True)
        elif "Tello" in self.folder_path:
            hec_rotation_matrix = Rotation.from_euler('xyz', [0, 190, 170], degrees=True) #cv2.Rodrigues(hec_angles)[0]
            # hec_rotation_matrix = Rotation.from_euler('xyz', [0, 180, -45], degrees=True)
        else:
            hec_rotation_matrix = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)

        hec_rotation_matrix = hec_rotation_matrix.as_matrix()
        vo_to_rotate = vo_to_rotate[:,:,np.newaxis]
        vo_rotated = hec_rotation_matrix@vo_to_rotate
        rotate_matrix_ = rotate_matrix
        estimated_path_3d = np.squeeze(vo_rotated, axis=2) 
        estimated_path_3d = np.copy(estimated_path_3d + gt_path_3d[0])
        estimated_rot_vec = []
        for i in range(len(rotate_matrix_)):
            R, _ = cv2.Rodrigues(rotate_matrix_[i]) 
            estimated_rot_vec.append((R[0][0],R[1][0],R[2][0]))
        estimated_rot_vec = estimated_rot_vec + gt_rot_vec[0]
        # estimated_path_3d[1:] = estimated_path_3d[1:] # + np.array([0,0,0.18])
       
        dist_gt = np.zeros(len(gt_path_3d))
        dist_gt[1:] = np.linalg.norm(gt_path_3d[1:] - gt_path_3d[:-1], axis=1)
        for i in range(1,len(dist_gt)):
            dist_gt[i] += dist_gt[i-1] 
        dist_vo = np.linalg.norm(estimated_path_3d, axis=1)
        errors = np.linalg.norm(gt_path_3d - estimated_path_3d, axis=1)
        error_per = 100* errors[1:] / dist_gt[1:]
        rmse = np.sqrt(np.mean(np.square(errors)))
        rmse_p = np.sqrt(np.mean(np.square(error_per)))
        std = np.std(errors)
        err_max = np.max(dist_gt)
        err_min = np.min(dist_gt)
        # per_rms = 100*rmse/(err_max-err_min)
        print("RMS Error:", "{: .4f}".format(rmse)+"m"," Mean Error:", "{: .4f}".format(np.mean(errors))+"m", "Standard devistion of errors:", "{: .4f}".format(std)+"m","% RMSE:", "{: .2f}".format(rmse_p)+"%")

       # Specify the file name
        # file_name = self.folder_path+"/error/Q/"+"angledynamics_error_1.npy"
        # file_name1 = self.folder_path+"/error/"+"750dynamics_%error_.npy"
        # gt = self.folder_path+"/error/"+"gt.npy"

        # Save the array to a file
        # np.save(file_name, errors)
        # np.save(file_name1, error_per)
        # np.save(gt, dist_gt)
        """-----------Plot the Results to view the estimated path alongside the ground truth----------"""
        # Create a figure and two subplots
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(estimated_path_3d[:,2],estimated_path_3d[:,0],estimated_path_3d[:,1],label='Visual Odometry')
        ax.plot(gt_path_3d[:,2],gt_path_3d[:,0],gt_path_3d[:,1],label='Ground Truth')
        #plt.grid()
        ax.legend()
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
        #plt.show()

        #plt.close('all')
        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the first graph on the first subplot
        ax1.plot(dist_gt, errors)
        ax1.set_title('Error measurment')
        ax1.set_ylabel("Error in m")
        ax1.set_xlabel("Ground truth in m")
        ax1.grid()

        # Plot the second graph on the second subplot
        ax2.plot(dist_gt[3:], error_per[2:])
        ax2.set_title('Error %')
        ax2.set_ylabel("Error in %")
        ax2.set_xlabel("Ground truth in m")
        ax2.grid()


        gig, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(12, 5))

        estimated_rot_vec = np.degrees(estimated_rot_vec)
        gt_rot_vec = np.degrees(gt_rot_vec)
        # errors = gt_rot - estimated_rot
        # Plot the first graph on the first subplot
        bx1.plot(dist_gt, gt_rot_vec[:,0], label='GT roll')#-estimated_rot[:,2])
        bx1.plot(dist_gt, estimated_rot_vec[:,0], label='VO roll')
        # bx1.set_title('Error measurment')
        bx1.set_ylabel("Rotation in degrees")
        bx1.set_xlabel("distance covered in m")
        bx1.grid()

        # Plot the second graph on the second subplot
        bx2.plot(dist_gt, gt_rot_vec[:,1], label='GT Pitch')
        bx2.plot(dist_gt, estimated_rot_vec[:,1], label='VO Pitch')
        # bx1.set_ylabel("Rotation in degree")
        # bx1.set_xlabel("distance covered in m")
        bx2.grid()
        bx2.legend()
        
        # Plot the second graph on the second subplot
        bx3.plot(dist_gt, gt_rot_vec[:,2], label='GT Yaw')
        bx3.plot(dist_gt, estimated_rot_vec[:,2], label='VO Yaw')
        # bx3.set_title('Error %')
        # bx3.set_ylabel("Error in %")
        # bx3.set_xlabel("Ground truth in m")
        bx3.grid()
        bx3.legend()
        
        '''
        # Plot the third graph on the second subplot
        ax3.plot(range(1,len(distance_error_z)+1), distance_error_z)
        ax3.set_title('Error in Z axis')

        ax4.plot(range(len(error_per)), error_per)
        ax4.set_title('Error percentage')'''

        hig, (cx1, cx2, cx3) = plt.subplots(1, 3, figsize=(12, 5))

        err =  gt_rot_vec[:,0]-estimated_rot_vec[:,0]
        gx,cx,pre_err = regression(dist_gt, err)
        cx1.plot(dist_gt,err,label='Error')
        cx1.plot(dist_gt,pre_err, label='Predicted Model')
        # cx1.set_title('Error measurment')
        cx1.set_ylabel("Rotation Error X in degrees")
        cx1.set_xlabel("distance covered in m")
        cx1.legend()
        cx1.grid()

        # Plot the second graph on the second subplot
        err =  gt_rot_vec[:,1]-estimated_rot_vec[:,1]
        gy,cy,pre_err = regression(dist_gt, err)
        cx2.plot(dist_gt,err,label='Error')
        cx2.plot(dist_gt,pre_err, label='Predicted Model')
        # cx2.set_title('Error measurment')
        cx2.set_ylabel("Rotation Error Y in degrees")
        # cx1.set_xlabel("distance covered in m")
        cx2.legend()
        cx2.grid()
        
        # Plot the second graph on the second subplot
        err =  gt_rot_vec[:,2]-estimated_rot_vec[:,2]
        gz,cz,pre_err = regression(dist_gt, err)
        cx3.plot(dist_gt,err,label='Error')
        cx3.plot(dist_gt,pre_err, label='Predicted Model')
        # cx3.plot(dist_gt, gt_rot_vec[:,2]-estimated_rot_vec[:,2])
        # cx1.set_title('Error measurment')
        cx3.set_ylabel("Rotation Error Z in degrees")
        # cx3.set_xlabel("distance covered in m")
        cx3.legend()
        cx3.grid()

        print(gx,cx,gy,cy,gz,cz)

        # Add labels and title to the entire figure
        fig.suptitle('Translation Error Measurements')
        hig.suptitle('Rotational Error Measurements ')
        gig.suptitle('Rotational Motion ')
        plt.show()
        

        # Update GUI after VO finishes
        # self.vo_finished()
        # self.root.after(0, self.vo_finished())

    def vo_finished(self):
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="VO Finished")
        root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = VOApp(root)
    root.mainloop()
