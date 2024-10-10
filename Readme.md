# Event-3DGS: Event-based 3D Reconstruction Using 3D Gaussian Splatting

## Introduction
This repository contains the research code for **Event-3DGS: Event-based 3D Reconstruction Using 3D Gaussian Splatting**. The code is designed to implement the event-based 3D reconstruction algorithm described in the paper and includes key components such as the photovoltage contrast estimation module and a novel event-based loss for optimizing reconstruction quality.

The code has been tested in a personal environment on Windows 11. If you encounter any difficult-to-resolve issues during deployment in other environments, please feel free to provide feedback.

## Installation
Follow the steps below to set up the environment and install dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lanpokn/Event-3DGS.git
   cd Event-3DGS
   ```

2. **Install the necessary dependencies:**

   This project is based on 3DGS (https://github.com/graphdeco-inria/gaussian-splatting), so please refer to its installation instructions. 

   Some parts of the code in this project use additional libraries, which were mainly my personal attempts during the exploratory phase and **can be ignored during use**.

## Dataset Format
To ensure proper usage, we will introduce the format in which we organize the data.

1. **Data organization :**

   - We organize the data into the following structure:
     ```
     /path/to/dataset/
       ├── training/
       ├── validation/
       ├── testing/
     ```

2. **Dataset Configuration:**
   - Update the paths in the configuration file `config/dataset_config.yaml` to point to the correct dataset location.

## Getting Started
Once you've set up the environment and downloaded the dataset, follow these steps to run the code:

1. **Train the model:**
   ```bash
   python train.py --config config/train_config.yaml
   ```

2. **Evaluate the model:**
   ```bash
   python evaluate.py --checkpoint /path/to/checkpoint --config config/eval_config.yaml
   ```

3. **Visualize the results:**
   ```bash
   python visualize.py --data /path/to/data
   ```

## Features
The repository includes several functionalities:
- **Event-based 3D reconstruction** using the proposed 3D Gaussian Splatting (3DGS) method.
- **Photovoltage contrast estimation** using a high-pass filter.
- **Novel event-based loss function** for optimizing 3D reconstruction quality.
- **Pre-trained models** for quick experimentation.

## Acknowledgments
We thank the authors of https://github.com/graphdeco-inria/gaussian-splatting and the other open-source libraries used in this work. 
