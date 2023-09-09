# Project Name

Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

The current repository provides the basic code for the deepmodulus (PINNs based material parameter identification code). The code aim to extract the elastic parameters of the hyperelastic material from the loading test and the digital image correlation.

## Features

The current code can help to extract the elastic modulus from the loading curve and digital image correlation.
- Figure-1
  
  ![Diagram of the PINNs](\Figure\Figure-1.png)

- Figure-2
  
  ![Forward_Problem](\Figure\Figure-2.png)

- Figure-3
  
  ![Inverse_Problem_without_DIC](\Figure\Figure-3.png)

- Figure-4
  
  ![Inverse_Problem_with_DIC](\Figure\Figure-4.png)

## Getting Started

Provide instructions on how to get started with your project. This should include information on prerequisites and installation.

### Prerequisites

The current code is developed based on Python 3.7.0. The main package needed is  Deepxde 1.7.0. which is available on the website https://deepxde.readthedocs.io/en/latest/user/installation.html. Please check it for the detailed instructions of installation. All the packages the author currently uses is given in the file "TF-2.5.yaml". For the best performance, please set the GPU environment for Tensorflow.

### Installation

Step 1:
Install the Deepxde 1.7.0.
Step 2:
Go to the repository of the deepxde. Replace the “booundary_conditions.py” with the boundary conditions provided. This new boundary_conditions added three more boundary conditions

## Usage

Here, we will provide an example of 
Provide examples or documentation on how to use your project. Include code snippets or screenshots if necessary.

## Contributing

Siyuan Song, Hanxun Jin developed this code in 2013.
Please cite:
Siyuan Song, Hanxun Jin. Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks. 2023

For any questions, please contact
songsiyuanxjtu@gmail.com


