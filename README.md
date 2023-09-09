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
  
  ![Diagram of the PINNs](/Figure/Figure-1.png)

## Getting Started

In the current section, I will present a quick example about how to run the Deepmodulus code.

### Prerequisites

The current code is developed based on Python 3.7.0. The main package needed is Deepxde 1.7.0. which is available on the website https://deepxde.readthedocs.io/en/latest/user/installation.html. Please check the website for the installation details. In addition, all the packages the author currently uses is given in the file "TF-2.5.yaml". Please install the GPU environment for tensorflow to accelerate the calculation.

### Installation

Step 1:
Install the Deepxde 1.7.0.
Step 2:
Go to the repository of the deepxde. Replace the “booundary_conditions.py” with the boundary conditions provided. 

## Usage

### For the pregiven calculation Case.
The current example provided here is based on the section 3.3 in the references. To run the code, please download the folder "Data", "Template" and the file "Run.py", and put them in the same folder. In "Run.py", you could set the noise value of speckle data, the number of epochs, the number of speckles and the number of tests you want to make.

### For any calculation case.
In hte real 


## Contributing

Siyuan Song, Hanxun Jin developed this code in 2013.
Please cite:
Siyuan Song, Hanxun Jin. Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks. 2023

For any questions, please contact
songsiyuanxjtu@gmail.com


