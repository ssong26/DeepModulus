# Project Name

Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks(v1.0.0)

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

The code is presently developed using Python 3.7.0 as its foundation, with the primary requirement being Deepxde 1.7.0, which can be found on the [official website](https://deepxde.readthedocs.io/en/latest/user/installation.html). Detailed installation instructions can be accessed on the website. Furthermore, the "TF-2.5.yaml" file provides a comprehensive list of all the packages employed by the author. To expedite calculations, it is recommended to install a GPU environment for TensorFlow.

### Installation

Step 1:
Install the Deepxde 1.7.0.
Step 2:
Go to the repository of the deepxde package. Replace the “boundary_conditions.py” with the new “boundary_conditions.py” provided. In the new file, we replace the "periodic boundary condition" to the "integration boundary condition" and the "Robin boundary condition" to the "operator boundary condition for the given pointset.

## Usage

The existing code is designed to work with a rectangular sample containing multiple spherical inhomogeneities. For a comprehensive understanding of its functionalities and capabilities, you can refer to the paper authored by Siyuan Song and Hanxun Jin in 2023, [Siyuan Song, Hanxun Jin (2023)](https://arxiv.org/pdf/2308.15640.pdf). Importantly, it is worth noting that the current code is easily adaptable to accommodate complex structures featuring arbitrary inhomogeneities.

### For the example.
The current example provided here is based on the section 3.3 in the references. To run the code, please download the folder "Data", "Template" and the file "Run.py", and put them in the same folder. In "Run.py", you could set the noise value of speckle data, the number of epochs, the number of speckles and the number of tests you want to make. The training speed is about 50 epochs/second. During the trainning, you could see a file "variable_history", which prints the material parameters as a function of the trainning epochs. 


## Contributing

Siyuan Song, Hanxun Jin developed this code in 2023. The website is currently maintained and updated by Siyuan Song.

Please cite:
Siyuan Song, Hanxun Jin. "Identifying Constitutive Parameters for Complex Hyperelastic Solids using Physics-Informed Neural Networks." arXiv preprint [arXiv:2308.15640 (2023)](https://arxiv.org/pdf/2308.15640.pdf)


For any questions, please contact
songsiyuanxjtu@gmail.com


