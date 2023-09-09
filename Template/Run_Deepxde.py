#noise = 0.00
#num_speckle = 200
#num_epochs = 1000000
#
import deepxde as dde
import numpy as np
from matplotlib import pyplot as plt
#
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.optimize import minimize
import pandas as pd
from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show,gca
import matplotlib.ticker as ticker
#
from deepxde.backend import tf
#
import random
import time
#
stretch = 2.5
#
force_data = pd.read_csv('../Data/Force.csv')
time_list = force_data["Time"]
force_list = force_data["Force"]
num_time = len(time_list)
num_fitting = 5
fitting_coeff = np.polynomial.polynomial.polyfit(time_list,force_list,num_fitting)
#
def force_history(tim):
    s = 0
    for i in range(num_fitting+1):
        s = s + fitting_coeff[i] * tim**i
    return s
#
#plt.plot(time_list,force_history(time_list))
#plt.scatter(time_list,force_list)
#
lam_list = np.array(time_list) * (stretch - 1.0) + 1.0
def fit_force_operator(s):
    #
    mu_final  = s[0]
    lam_final = s[1]
    #
    F11 = lam_list
    F12 = 0.0
    F21 = 0.0
    F22 = 1/np.sqrt(lam_list)
    F33 = 1/np.sqrt(lam_list)
    #
    I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_final**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_final**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_final**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_final**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    return np.max(np.abs(P11 * mu_final - force_list))
#
trial_para = minimize(fit_force_operator,[1,3]).x
#
def transform(x, y):
    #
    X1 = x[:, 0:1]
    X2 = x[:, 1:2]
    t  = x[:, 2:3]
    ratio = (stretch - 1.0) * t + 1
    ratio_inv = 1/ratio
    new_x1 = (X1 + 1.0) * (X1 - 1.0) * y[:, 0:1] + (X1 + 1.0) * ratio - 1.0
    new_x2 = (X1 + 1.0) * (X1 - 1.0) * y[:, 1:2] + X2
    return tf.concat([new_x1,new_x2,y[:, 2:3]], axis=1)
#
x_force_list = []
y_force_list = []
num_points_force = 400
for i in range(num_points_force):
    x = (np.random.random()*0.2 + 0.6) * (float(np.random.random() > 0.5) * 2.0 - 1.0)
    y = np.random.random() - 0.5
    x_force_list.append(x)
    y_force_list.append(y)
x_force_list = np.array(x_force_list)
y_force_list = np.array(y_force_list)
#
points_force = []
value_force = []
for i in range(num_time):
    t = time_list[i]
    force = force_list[i]
    for j in range(len(x_force_list)):
        points_force.append([x_force_list[j],y_force_list[j],t])
        value_force.append([force])
#
points_force = np.array(points_force)
value_force  = np.array(value_force)
#
num_points_upper_lower = 100
dx = 2/num_points_upper_lower
upper_lower_list = np.linspace(-1.0+dx/2,1.0-dx/2,num_points_upper_lower)
#
points_upper_lower = []
value_upper_lower = []
for i in range(num_time):
    t = time_list[i]
    for j in range(len(upper_lower_list)):
        points_upper_lower.append([upper_lower_list[j],-0.5,t])
        value_upper_lower.append([0.0])
        points_upper_lower.append([upper_lower_list[j],0.5,t])
        value_upper_lower.append([0.0])
#
points_upper_lower = np.array(points_upper_lower)
value_upper_lower  = np.array(value_upper_lower)
#
def calculate_force_operator(x, y):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_final**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_final**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_final**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_final**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    return P11 * mu_final
#
def calculate_force_inner_operator(x, y):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_final**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_final**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_final**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_final**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    n1 = x[:,0:1]
    n2 = x[:,1:2]
    return (n1 * P11 + n2 * P12)**2 + (n1 * P21 + n2 * P22)**2
#
def calculate_force_upper_lower_operator(x, y):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_final**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_final**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_final**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_final**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    return (P12)**2 + (P22)**2
#
def pde_operator(x, y):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_final**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_final**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_final**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_final**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    BoF1 = dde.grad.jacobian(P11, x, i=0, j=0) + dde.grad.jacobian(P12, x, i=0, j=1)
    BoF2 = dde.grad.jacobian(P21, x, i=0, j=0) + dde.grad.jacobian(P22, x, i=0, j=1)
    #
    incom = (F22*F11 - F21*F12)*F33 - 1
    #
    return [BoF1,BoF2,incom]
#

# Step-2
# Compare the displacement field. Let the two deformed shape together.
import math
num_time = 21
folder_address = "../Data/"
dic_node= np.load(folder_address+"dic_node.npy",allow_pickle=True,encoding='bytes').item()
dic_connectivity = np.load(folder_address+"dic_connectivity.npy",allow_pickle=True,encoding='bytes').item()
dic_displacement = []
for i in range(num_time):
    displacement= np.load(folder_address + str(i)+"_dic_displacement.npy",allow_pickle=True,encoding='bytes').item()
    dic_displacement.append(displacement)
# Find the Boundary
dic_bc_upper = []
dic_bc_right = []
dic_bc_lower = []
dic_bc_left  = []
dic_bc_inner = [] # Means the point near the hole.
for index in dic_node:
    x = dic_node[index][0]
    y = dic_node[index][1]
    if x == -1:
        dic_bc_left.append([index,x,y])
    if x == 1:
        dic_bc_right.append([index,x,y])
    if y == -0.5:
        dic_bc_lower.append([index,x,y])
    if y == 0.5:
        dic_bc_upper.append([index,x,y])
    # Hole_1
    hole = np.array([[-0.4,0.3],
                     [-0.2,-0.1],
                     [0.1,0.2],
                     [0.3,-0.3]])
    #
    dist = np.linalg.norm(np.array(dic_node[index]) - hole,axis = 1) - 0.15
    if np.sum(dist < -(10**-4)) > 0:
        continue
    #
    dic_bc_inner.append([index,x,y])
    
#
dic_bc_upper = np.array(dic_bc_upper)
dic_bc_right = np.array(dic_bc_right)
dic_bc_lower = np.array(dic_bc_lower)
dic_bc_left  = np.array(dic_bc_left)

dic_bc_upper = dic_bc_upper[np.argsort(dic_bc_upper[:,1]),:]
dic_bc_right = dic_bc_right[np.argsort(dic_bc_right[:,2])[::-1],:]
dic_bc_lower = dic_bc_lower[np.argsort(dic_bc_lower[:,1])[::-1],:]
dic_bc_left  = dic_bc_left[np.argsort(dic_bc_left[:,2]),:]
#
import random
dic_bc_inner = random.sample(dic_bc_inner,num_speckle)
#
dic_bc_inner  = np.array(dic_bc_inner)
#
time_list = np.linspace(0,1,num_time)
# Reorder
x_new_list = []
y_new_list = []
coord_old_list = []
x_new_inner_list = []
y_new_inner_list = []
coord_old_inner_list = []
for i in range(num_time):
    x_new_list.append([])
    y_new_list.append([])
    x_new_inner_list.append([])
    y_new_inner_list.append([])
    coord_old_list.append([])
    coord_old_inner_list.append([])
    for ele_list in [dic_bc_upper,dic_bc_right,dic_bc_lower,dic_bc_left]:
        for ele in ele_list:
            index = int(ele[0])
            x_new_list[-1].append( ele[1] + dic_displacement[i][index][0] ) 
            y_new_list[-1].append( ele[2] + dic_displacement[i][index][1] )
            coord_old_list[-1].append([ele[1],ele[2],time_list[i]])
    #
    for ele in dic_bc_inner:
        index = int(ele[0])
        #
        err_x = noise * np.random.random() * (float(np.random.random()>0.5) * 2 - 1.0) + 1.0
        err_y = noise * np.random.random() * (float(np.random.random()>0.5) * 2 - 1.0) + 1.0
        #
        x_new_inner_list[-1].append( ele[1] + dic_displacement[i][index][0] * err_x ) 
        y_new_inner_list[-1].append( ele[2] + dic_displacement[i][index][1] * err_y )
        #
        coord_old_inner_list[-1].append([ele[1],ele[2],time_list[i]])
#
x_new_list = np.array(x_new_list)
y_new_list = np.array(y_new_list)
coord_old_list = np.array(coord_old_list)
x_new_inner_list = np.array(x_new_inner_list)
y_new_inner_list = np.array(y_new_inner_list)
coord_old_inner_list = np.array(coord_old_inner_list)
# 
# 
#
dim = coord_old_inner_list.shape
inner_speckle_list = coord_old_inner_list.reshape([dim[0]*dim[1],3])
#
dim = x_new_inner_list.shape
inner_speckle_list_value_x = x_new_inner_list.reshape([dim[0]*dim[1],1])
dim = y_new_inner_list.shape
inner_speckle_list_value_y = y_new_inner_list.reshape([dim[0]*dim[1],1])
#
net = dde.maps.FNN([3] + [30]*3 + [3], "tanh", "Glorot uniform")
net.apply_output_transform(transform)
#
#
outer = dde.geometry.geometry_2d.Rectangle([-1.0,-0.5],[1.0,0.5])
inner_0 = dde.geometry.geometry_2d.Disk(hole[0],0.15)
inner_1 = dde.geometry.geometry_2d.Disk(hole[1],0.15)
inner_2 = dde.geometry.geometry_2d.Disk(hole[2],0.15)
inner_3 = dde.geometry.geometry_2d.Disk(hole[3],0.15)
spatial_domain = outer - inner_0 - inner_1 - inner_2 - inner_3
temporal_domain = dde.geometry.TimeDomain(0.0, 1.0)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)
#
def pde(x, y):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    incom = (F22*F11 - F21*F12)*F33 - 1
    #
    I1_real = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    LAM = (stretch - 1.0) * x[:, 2:3] + 1.0
    #
    I1_imag = LAM * LAM + 1/LAM + 1/LAM
    #
    I1 = I1_imag * coeff + I1_real * (1.0 - coeff)
    #
    lam = lam_imag * coeff_lam + lam_load * ( 1.0 - coeff_lam )
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    BoF1 = dde.grad.jacobian(P11, x, i=0, j=0) + dde.grad.jacobian(P12, x, i=0, j=1)
    BoF2 = dde.grad.jacobian(P21, x, i=0, j=0) + dde.grad.jacobian(P22, x, i=0, j=1)
    return [incom,BoF1,BoF2]
#
def boundary_upper_lower(x, on_boundary):
    return on_boundary and np.isclose(np.abs(x[1]), 0.5)  and (not np.isclose(np.abs(x[0]), 1.0))
#
def bc_func(x, y, X):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    incom = (F22*F11 - F21*F12)*F33 - 1
    #
    I1_real = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    LAM = (stretch - 1.0) * x[:, 2:3] + 1.0
    #
    I1_imag = LAM * LAM + 1/LAM + 1/LAM
    #
    I1 = I1_imag * coeff + I1_real * (1.0 - coeff)
    #
    lam = lam_imag * coeff_lam + lam_load * ( 1.0 - coeff_lam )
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33 
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    return tf.sqrt(P12**2 + P22**2)
#
def bc_func_inner(x, y, X):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    incom = (F22*F11 - F21*F12)*F33 - 1
    #
    I1_real = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    LAM = (stretch - 1.0) * x[:, 2:3] + 1.0
    #
    I1_imag = LAM * LAM + 1/LAM + 1/LAM
    #
    I1 = I1_imag * coeff + I1_real * (1.0 - coeff)
    #
    lam = lam_imag * coeff_lam + lam_load * ( 1.0 - coeff_lam )
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    n1 = x[:,0:1]
    n2 = x[:,1:2]
    return tf.sqrt((n1 * P11 + n2 * P12)**2 + (n1 * P21 + n2 * P22)**2)/0.15
#
def func_total_force(x, y, X):
    #
    F33   = y[:,2:3]
    F11 = dde.grad.jacobian(y, x, i=0, j=0)
    F12 = dde.grad.jacobian(y, x, i=0, j=1)
    F21 = dde.grad.jacobian(y, x, i=1, j=0)
    F22 = dde.grad.jacobian(y, x, i=1, j=1)
    #
    incom = F22*F11 - F21*F12 - 1
    #
    I1_real = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33
    #
    LAM = (stretch - 1.0) * x[:, 2:3] + 1.0
    I1_imag = LAM * LAM + 1/LAM + 1/LAM
    #
    I1 = I1_imag * coeff + I1_real * (1.0 - coeff)
    #
    coe0 = 2.0 * 0.5
    coe1 = 2.0 * 2.0/(20.0*lam_load**2)
    coe2 = 2.0 * 3.0 * 11.0/(1050*lam_load**4)
    coe3 = 2.0 * 4.0 * 19.0/(7000*lam_load**6)
    coe4 = 2.0 * 5.0 * 519/(673750*lam_load**8)
    coe = (coe0+coe1*I1+coe2*(I1**2)+coe3*(I1**3)+coe4*(I1**4))
    #
    #p = coe*F33/(F11*F22 - F12*F21) 
    p = coe * F33 * F33
    #
    P11 = (-p*F22*F33 + coe*F11)
    P12 = ( p*F21*F33 + coe*F12)
    P21 = ( p*F12*F33 + coe*F21)
    P22 = (-p*F11*F33 + coe*F22)
    #
    return P11 * mu

"""
def boundary_inner(x, on_boundary):
    #print(x.shape)
    return on_boundary and np.isclose(x[0] ** 2 + x[1] ** 2, 0.15**2)
"""
#
def boundary_speckle_x(x, y, X):
    return y[:,0:1]
def boundary_speckle_y(x, y, X):
    return y[:,1:2]
#
#bc_upper_lower = dde.OperatorBC(spatio_temporal_domain,bc_func,boundary_upper_lower)
bc_upper_lower = dde.RobinBC(points_upper_lower,bc_func,value_upper_lower)
#
#bc_inner = dde.OperatorBC(spatio_temporal_domain,bc_func_inner,boundary_inner)
#bc_inner = dde.RobinBC(points_hole,bc_func_inner,value_hole)
#
bc_right_force = dde.PeriodicBC(points_force,func_total_force,value_force,num_time,num_points_force)
#
bc_speckle_x   = dde.RobinBC(inner_speckle_list,boundary_speckle_x,inner_speckle_list_value_x)
bc_speckle_y   = dde.RobinBC(inner_speckle_list,boundary_speckle_y,inner_speckle_list_value_y)
#
data = dde.data.TimePDE(spatio_temporal_domain, pde, [bc_upper_lower,bc_speckle_x,bc_speckle_y,bc_right_force], num_domain=20000, num_boundary=0,num_initial=0, num_test=2000)
#
model = dde.Model(data, net)
#
weights = [1 for i in range(7)]
weights[0]  = 10 # Incom
weights[3]  = 10 # UpperLower
weights[4]  = 1000 # Speckle-x
weights[5]  = 400 # Speckle-y
weights[6] = 100
#
# Step-1 Train NeoHookean
#
mu = dde.Variable(4.0) #truth 1.0
#mu = dde.Variable(trial_para[0])
coeff = 0.0
coeff_lam = 0.0
#lam_imag = trial_para[1]
lam_imag = 7.0
lam_load = dde.Variable(7.0)
#lam_load = dde.Variable(trial_para[1])
#
model.compile("adam", lr=0.001,external_trainable_variables=[mu,lam_load],loss_weights = weights)
variable = dde.callbacks.VariableValue([mu,lam_load], period=100,filename="variable_history",precision=9)
#model.train(epochs=50000,display_every = 1000, callbacks=[variable])
#
model.train(epochs=num_epochs,display_every = 1000, callbacks=[variable])
model.save("Siyuan_AUG_12")
#
#
np.save("train_x.npy",data.train_x)
np.save("train_x_all.npy",data.train_x_all)
np.save("train_x_bc.npy",data.train_x_bc)
np.save("test_x.npy",data.test_x)
#
#
np.save("steps.npy",model.losshistory.steps)
np.save("loss_train.npy",model.losshistory.loss_train)
np.save("loss_test.npy",model.losshistory.loss_test)
np.save("loss_weights.npy",model.losshistory.loss_weights)
#