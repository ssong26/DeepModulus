# Project Name

Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks
v1.0.0

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

The present repository offers foundational code for deepmodulus, a codebase centered on the use of Physics-Informed Neural Networks (PINNs) for material parameter identification. Its primary objective is to facilitate the extraction of elastic parameters governing hyperelastic materials by leveraging loading tests and digital image correlation.


## Features

The existing code serves as a valuable tool for extracting the elastic modulus of hyperelastic materials through the analysis of both the loading curve and digital image correlation.
- Figure-1
  
  ![Diagram of the PINNs](/Figure/Figure-1.png)

## Getting Started

In this section, I will provide a brief example demonstrating the execution of the Deepmodulus code.

### Prerequisites

The current code is developed based on Python 3.7.0. The main package needed is Deepxde 1.7.0. which is available on the [website](https://deepxde.readthedocs.io/en/latest/user/installation.html). Please check the website for the installation details. In addition, all the packages the author currently uses is given in the file "TF-2.5.yaml". Please install the GPU environment for tensorflow to accelerate the calculation.

### Installation

Step 1:
Install the Deepxde 1.7.0.
Step 2:
Go to the repository of the deepxde. Replace the “booundary_conditions.py” with the boundary conditions provided. 

## Usage

The current code is designed for the rectangular sample with spherical inhomogeneities. However, it can be quickly extended to the complex structures with arbitrary inhomogeneities. To run this code, we need know the constitutive relations, the loading curve and the speckle information. Fpr the first one, we need the balance of force and boundary conditions. The latter two terms can either be obtained from the real experiment or from the FEM simulations.

### For the example.
The current example provided here is based on the section 3.3 in the references. To run the code, please download the folder "Data", "Template" and the file "Run.py", and put them in the same folder. In "Run.py", you could set the noise value of speckle data, the number of epochs, the number of speckles and the number of tests you want to make. The training speed is about 50 epochs/second. During the trainning, you could see a file "variable_history", which prints the material parameters as a function of the trainning epochs. 


## Contributing

Siyuan Song, Hanxun Jin developed this code in 2023. The website is currently maintained and updated by Siyuan Song.

Please cite:
Siyuan Song, Hanxun Jin. "Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks." arXiv preprint [arXiv:2308.15640 (2023)](https://arxiv.org/pdf/2308.15640.pdf)


For any questions, please contact
songsiyuanxjtu@gmail.com


