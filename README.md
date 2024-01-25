# Monocular (Inertial) Visual Odometry for Drones

This project focuses on developing a monocular (inertial) visual odometry system for drone navigation. The goal is to provide accurate and robust localization capabilities using only visual and inertial sensor data.


## Overview

Monocular (inertial) visual odometry is a technique that combines information from a single camera and inertial sensors to estimate the drone's motion and position. This repository implements a system that leverages these sensors to achieve reliable and precise navigation for drones.


## Installation

Follow these steps to install and set up the monocular (inertial) visual odometry system:

1. Clone the repository: `git clone https://github.com/Yacynte/Monocular-Visual-Odometry.git`
2. Install dependencies: OpenCV, numpy, matplotlib, tqdm `pip install -r requirements.txt`


[Detailed Installation Guide](./docs/installation.md)


## Usage

To use the monocular (inertial) visual odometry system with your drone, follow these steps:

1. Configure the system parameters in `config.yaml`.
2. Provide input images and inertial sensor data.
3. Run the main script: `python visualodometry.py`

[Example Usage](./docs/usage.md)


## Configuration

Customize the behavior of the system by modifying the parameters in the `config.yaml` file. This file includes settings such as camera calibration, feature extraction thresholds, and filtering parameters.

[Configuration Details](./docs/configuration.md)


## Data Format

The monocular (inertial) visual odometry system supports the following data formats:

- Image formats: JPEG, PNG
- Inertial sensor data: CSV format
- Ground truth data: CSV, TXT

[Input Data Guidelines](./docs/data_format.md)


## Results

Here are some visualizations demonstrating the results obtained by the monocular (inertial) visual odometry system:

<<<<<<< HEAD
- [Result Image 1](./results/image_1.png)
- [Result Image 2](./results/image_2.png)
=======
- [Result Image 1](./results/image_2.png)
- [Result Image 2](./results/image_1.png)
>>>>>>> a60ebe420cfdef9123106c94b3f60cc906d02c36

