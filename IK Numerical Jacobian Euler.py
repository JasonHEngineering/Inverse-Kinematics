# -*- coding: utf-8 -*-
"""
@Purpose: Inverse Kinematics (vector formula Jacobian) - Euler
@author: Jason H + Engineering
More explanation @website: https://jashuang1983.wordpress.com/inverse-kinematics-robotics-numerical-jacobian/
 
"""
import pandas as pd
import random
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from skspatial.objects import Plane, Point, Vector
import math
import plotly.io as pio
from math import (
    asin, pi, atan2, cos
)
from scipy.spatial.transform import Slerp, Rotation
from scipy.spatial.transform import Rotation as R
import time
from datetime import datetime
 
now = datetime.now()
dt_string = now.strftime("D"+"%d%m%Y"+"_T"+"%H%M%S")
csv_name = r'IK_logger_'+dt_string+'.csv'
 
data = []
column_headers = ['Rows',
                  'X', 'Y', 'Z',
                  'Pos Error',
                  'Roll', 'Pitch', 'Yaw',
                  'IK Iterations',
                  'IK Timestamp', 'Solve Status',
                  'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6',
                  'time_stay']
 
start_time = time.time()
 
pio.renderers.default='browser'
 
figure_axis_limit = 600
 
 
global desired_orientation
 
displacement_err = 0.01 # acceptable error for IK in mm
orientation_err = 0.01 # acceptable error for orientation in deg
 
lower_limit = [-170, -90, -90, -150, -90, -180]
upper_limit = [170, 90, 90, 150, 95, 180]
 
hinge_length = 24 # for illustrating the hinge in 3d view
max_iter = 3000 # iterations allow for IK CCD
step_size=0.1 #step size for jacobian delta_theta
error_list = []
orientation_error_list = []
 
index_of_angles = [2, 9, 12, 17, 23, 29] #hinge for project Hayley v3
local_linkage_data = [
    [0,0,0,0,0], #0
    [1,0,0,54,0], #1
    [1,0,0,0,0], #theta_1 - Axis1, anti-clockwise with positive theta #2
    [2,0,-90,0,0], #3
    [2,0,0,35,0], #4
    [2,0,90,0,0], #5
    [2,0,0,35.5,0], #6
    [2,0,90,0,0], #7
    [2,90,0,0,0], #8
    [2,0,0,0,0], #theta_2 - Axis2, down with positive theta #9
    [2,0,-180,0,0], #10
    [3,0,0,0,60], #11
    [3,0,0,0,0], #theta_3 - Axis3, down with positive theta #12
    [4,0,0,0,33],     #13
    [4,0,0,-34.5,0],  # from here, switch between X and Z axis #14
    [4,90,0,0,0],     #15
    [4,0,90,0,0],     #16
    [4,0,0,0,0], #theta_4 - Axis4, anti-clockwise from top view with positive theta  #17
    [5,0,0,54,0], #18
    [5,0,-90,0,0],     #19
    [5,-90,0,0,0],     #20
    [5,0,0,32,0], #21
    [5,0,0,0,29],  # from here, switch between X and Z axis #22
    [5,0,0,0,0], #theta_5 - Axis5, down with positive theta #23
    [6,0,0,0,47],#24
    [6,0,0,-32,0], #25
    [6,0,0,0,25], #26
    [6,90,0,0,0],     #27
    [6,0,90,0,0],     #28
    [6,0,0,0,0], #theta_6 - Axis6, anti-clockwise with positive theta #29
    [7,0,0,22,0], # dummy 20 mm extension #30
    [7,0,-90,0,0], # dummy 20 mm extension #31
    [7,-90,0,0,0], # dummy 20 mm extension #32
    ]
 
index_of_hinges = index_of_angles.copy()
 
list_of_thetas = [0,
  0,
  10.6, #Axis 1
  0,
  0,
  0,
  0,
  0,
  -90,
  24, #Axis 2
  0,
  0,
  85.4, #Axis 3
  0,
  0,
  90,
  0,
  -22.3, #Axis 4
  0,
  0,
  -90,
  0,
  0,
  57, #Axis 5
  0,
  0,
  0,
  90,
  0,
  137, #Axis 6
  0,
  0,
  -90]
 
angle_max = list_of_thetas.copy()
angle_min = list_of_thetas.copy()
list_of_blockers = list_of_thetas.copy()
 
for i, value in enumerate(index_of_angles):
    angle_max[value] = upper_limit[i]
    angle_min[value] = lower_limit[i]
    list_of_blockers[value] = 1
 
 
# special provision for the blockers to have theta to rotate the coodinate frame that is not due to hinge i.e theta due to mechanical link
for i, linkage in enumerate(local_linkage_data):
    if linkage[1] != 0:
        list_of_blockers[i] = 2
        list_of_thetas[i] = local_linkage_data[i][1]
 
def interpolate_two_points(points, num_interpolations):
    """
    Interpolate between two 3D points using NumPy's linspace.
 
    Parameters:
    - point1: Tuple of (x1, y1, z1)
    - point2: Tuple of (x2, y2, z2)
    - num_interpolations: Number of interpolations between the two points.
 
    Returns:
    - List of interpolated points as tuples [(x1, y1, z1), ..., (xn, yn, zn)]
    """
    x_values = np.linspace(points[0][0], points[1][0], num_interpolations)
    y_values = np.linspace(points[0][1], points[1][1], num_interpolations)
    z_values = np.linspace(points[0][2], points[1][2], num_interpolations)
 
    interpolated_points = list(zip(x_values, y_values, z_values))
    return interpolated_points
 
def rotation_matrix_to_euler(orientation_matrix):
 
    R11 = orientation_matrix[0,0]
    R12 = orientation_matrix[0,1]
    R13 = orientation_matrix[0,2]
 
    R21 = orientation_matrix[1,0]
    R22 = orientation_matrix[1,1]
    R23 = orientation_matrix[1,2]
 
    R31 = orientation_matrix[2,0]
    R32 = orientation_matrix[2,1]
    R33 = orientation_matrix[2,2]
 
    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
 
    if round(R31,4) != 1.0000 and round(R31,4) != -1.0000:
        #print(R31)
        pitch_1 = -1*asin(R31)
        pitch_2 = pi - pitch_1
        roll_1 = atan2( R32 / cos(pitch_1) , R33 /cos(pitch_1) )
        roll_2 = atan2( R32 / cos(pitch_2) , R33 /cos(pitch_2) )
        yaw_1 = atan2( R21 / cos(pitch_1) , R11 / cos(pitch_1) )
        yaw_2 = atan2( R21 / cos(pitch_2) , R11 / cos(pitch_2) )
 
         # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
         # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info).
        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
         yaw = 0 # anything (we default this to zero)
         if R31 == -1:
            pitch = pi/2
            roll = yaw + atan2(R12,R13)
         else:
            pitch = -pi/2
            roll = -1*yaw + atan2(-1*R12,-1*R13)
 
    # convert from radians to degrees
    roll = roll*180/pi
    pitch = pitch*180/pi
    yaw = yaw*180/pi
 
    rxyz_deg = np.array([roll , pitch , yaw])
 
    return rxyz_deg
 
def DH_matrix(theta, alpha, delta, rho):
 
    transient_matrix = np.eye(4)
 
    # Handle 3d DH parameters, row-by-row, left-to-right
    theta_rad = theta/180*np.pi
    alpha_rad = alpha/180*np.pi
 
    transient_matrix[0,0]=np.cos(theta_rad)
    transient_matrix[0,1]=-np.sin(theta_rad)
    #transient_matrix[0,2]=0
    transient_matrix[0,3]=rho
 
    transient_matrix[1,0]=np.sin(theta_rad)*np.cos(alpha_rad)
    transient_matrix[1,1]=np.cos(theta_rad)*np.cos(alpha_rad)
    transient_matrix[1,2]=-np.sin(alpha_rad)
    transient_matrix[1,3]=-np.sin(alpha_rad) * delta
 
    transient_matrix[2,0]=np.sin(theta_rad)*np.sin(alpha_rad)
    transient_matrix[2,1]=np.cos(theta_rad)*np.sin(alpha_rad)
    transient_matrix[2,2]=np.cos(alpha_rad)
    transient_matrix[2,3]=np.cos(alpha_rad) * delta
 
    return transient_matrix
 
 
def input_linkage_angles(list_of_thetas):
 
    for i in range(len(list_of_thetas)):
        local_linkage_data[i][1] = list_of_thetas[i]
 
    array_matrix = []
    transformation_matrix = None
 
    for i, linkage in enumerate(local_linkage_data):
 
        # Rotations first
        transient_rotation = DH_matrix(linkage[1], linkage[2], 0, 0)
 
        if transformation_matrix is None:
            transformation_matrix = transient_rotation
        else:
            transformation_matrix = np.matmul(transformation_matrix, transient_rotation)
 
        # then the translations
        transient_translation = DH_matrix(0, 0, linkage[3], linkage[4])
 
        if transformation_matrix is None:
            transformation_matrix = transient_translation
        else:
            transformation_matrix = np.matmul(transformation_matrix, transient_translation)
 
        array_matrix.append(transformation_matrix)
 
    # if pose == True:
    #     transformation_matrix = np.matmul(transformation_matrix, orientation_matrix)
    #     array_matrix.append(transformation_matrix)
 
    return(array_matrix)
 
 
def sqrt_sum_aquare(input_list):
    sum_square = 0
    for value in input_list:
        sum_square += value*value
    return(math.sqrt(sum_square))
 
 
 
 
 
 
def Inverse_Kinematics_Jacobian_numerical(count):
 
    solved = False
 
    # print(count)
    # print(desired_orientation)
 
    err_end_to_target = math.inf
    minimum_error = math.inf
    epsilon = 1e-6 # a small value to perturb, assume this is in radians
 
 
    num_dimensions = 6  # first 3 are for X,Y,Z and the remaining 3 for euler XYZ representation
    num_joints = 6 # 6 joints
    jacobian_matrix = np.zeros([num_dimensions, num_joints])
 
    # for t in range(6):
    #     jacobian_matrix[5,t] = 1
 
    z_vector = [None]*6
    end_effector_to_current_joint = [None]*6
    jacobian_array = [None]*6
 
    for loop in range(max_iter):
 
        P = input_linkage_angles(list_of_thetas) # forward kinematics
        # P is an array of transformation matrix
        # adding on ... the array of matrix are for convenience of plotly traces later on
        # adding on ... IK itself does require to access the individual joint coordinates
        end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
        err_end_to_target = sqrt_sum_aquare(end_to_target)
 
        current_orientation = rotation_matrix_to_euler(P[-1][:3, :3])
        #print(current_orientation)
        #print(desired_orientation)
        error_orientation = desired_orientation - current_orientation
        abs_error_orientation = [abs(ele) for ele in error_orientation]
        # print(abs_error_orientation)
        # print((np.array(abs_error_orientation) < orientation_err).all())
        error_list.append([loop, err_end_to_target])
        orientation_error_list.append([loop, error_orientation])
 
        # record the angles of the best minimal error so far; yes the error can increase in further iterations
        if err_end_to_target < minimum_error:
            minimum_error = err_end_to_target
            least_error_angles = list_of_thetas.copy()
 
        if (err_end_to_target < displacement_err) and (np.array(abs_error_orientation) < orientation_err).all():
            solved = True
            break
        else:
 
            for i in range(num_dimensions):
                for k, value in enumerate(index_of_angles): # work on the numerical Jacobian
 
                    perturbed_theta = list_of_thetas.copy()
                    perturbed_theta[value] = (perturbed_theta[value] + epsilon)
 
                    perturbed_FK = input_linkage_angles(perturbed_theta) # forward kinematics, output is array of transformation matrices, already chain-multiplied from the individual link transformation
                    perturbed_position_delta = perturbed_FK[-1][:3, 3] - P[-1][:3, 3] # get positional delta due to perturb (concept of partial differentiation)
 
                    perturbed_orientation_delta = rotation_matrix_to_euler(perturbed_FK[-1][:3, :3]) - current_orientation
 
                    perturbed_delta_combined = np.concatenate([perturbed_position_delta, perturbed_orientation_delta])
 
                    # Numerical differentiation to compute the Jacobian entry
                    jacobian_matrix[i, k] = perturbed_delta_combined[i] / (epsilon)
 
 
 
 
            # for k, value in enumerate(index_of_angles):
            #     z_vector[k] = np.array(P[value][:3, 2])
            #     end_effector_to_current_joint[k] = np.transpose(np.array(P[-1][:3, 3] - P[value][:3, 3]))
            #     jacobian_array[k] = np.transpose(np.cross(z_vector[k],end_effector_to_current_joint[k]))
 
            #     for j in range(3):
            #         jacobian_matrix[j,k] = jacobian_array[k][j]
            #     for j in range(3,6):
            #         jacobian_matrix[j,k] = z_vector[k][j-3]
 
            end_to_target_array = np.transpose(np.append(np.array(end_to_target), error_orientation))
            delta_theta = np.linalg.pinv(jacobian_matrix).dot(end_to_target_array)  # Pseudo-inverse used here
            delta_theta = delta_theta*step_size
 
            # Update current joint angle values
            for k, value in enumerate(index_of_angles):
                list_of_thetas[value] = (list_of_thetas[value] + (delta_theta[k]))
                list_of_thetas[value] = (list_of_thetas[value] + 180) % 360 - 180
 
                # clamp angle
                if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]-(10*np.random.random(1)[0])
                if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]+(10*np.random.random(1)[0])
 
 
        # if solved:
        #     break
 
    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics
 
    return P, list_of_thetas, err_end_to_target, solved, loop
 
 
def Inverse_Kinematics_Jacobian(count):
 
    solved = False
 
    # print(count)
    # print(desired_orientation)
 
    err_end_to_target = math.inf
    minimum_error = math.inf
 
    jacobian_matrix = np.zeros([6,6])
 
    # for t in range(6):
    #     jacobian_matrix[5,t] = 1
 
    z_vector = [None]*6
    end_effector_to_current_joint = [None]*6
    jacobian_array = [None]*6
 
    for loop in range(max_iter):
 
        P = input_linkage_angles(list_of_thetas) # forward kinematics
        # P is an array of transformation matrix
        # adding on ... the array of matrix are for convenience of plotly traces later on
        # adding on ... IK itself does require to access the individual joint coordinates
        end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
        err_end_to_target = sqrt_sum_aquare(end_to_target)
 
        current_orientation = rotation_matrix_to_euler(P[-1][:3, :3])
        #print(current_orientation)
        #print(desired_orientation)
        error_orientation = desired_orientation - current_orientation
        abs_error_orientation = [abs(ele) for ele in error_orientation]
        # print(abs_error_orientation)
        # print((np.array(abs_error_orientation) < orientation_err).all())
        error_list.append([loop, err_end_to_target])
        orientation_error_list.append([loop, error_orientation])
 
        # record the angles of the best minimal error so far; yes the error can increase in further iterations
        if err_end_to_target < minimum_error:
            minimum_error = err_end_to_target
            least_error_angles = list_of_thetas.copy()
 
        if (err_end_to_target < displacement_err) and (np.array(abs_error_orientation) < orientation_err).all():
            solved = True
            break
        else:
            for k, value in enumerate(index_of_angles):
                z_vector[k] = np.array(P[value][:3, 2])
                end_effector_to_current_joint[k] = np.transpose(np.array(P[-1][:3, 3] - P[value][:3, 3]))
                jacobian_array[k] = np.transpose(np.cross(z_vector[k],end_effector_to_current_joint[k]))
 
                for j in range(3):
                    jacobian_matrix[j,k] = jacobian_array[k][j]
                for j in range(3,6):
                    jacobian_matrix[j,k] = z_vector[k][j-3]
 
            end_to_target_array = np.transpose(np.append(np.array(end_to_target), error_orientation))
            delta_theta = np.linalg.pinv(jacobian_matrix).dot(end_to_target_array)  # Pseudo-inverse used here
            delta_theta = delta_theta*step_size
 
            # Update current joint angle values
            for k, value in enumerate(index_of_angles):
                list_of_thetas[value] = (list_of_thetas[value] + (delta_theta[k]) * 180 / math.pi)
                list_of_thetas[value] = (list_of_thetas[value] + 180) % 360 - 180
 
                # clamp angle
                if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]-(10*np.random.random(1)[0])
                if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]+(10*np.random.random(1)[0])
 
 
        # if solved:
        #     break
 
    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics
 
    return P, list_of_thetas, err_end_to_target, solved, loop
 
def Inverse_Kinematics_CCD():
    solved = False
    err_end_to_target = math.inf
    minimum_error = math.inf
 
    for loop in range(max_iter):
        for i in range(len(local_linkage_data)-1, -1, -1):
 
            if list_of_blockers[i] != 0:
 
                P = input_linkage_angles(list_of_thetas) # forward kinematics
                # P is an array of transformation matrix
                # adding on ... the array of matrix are for convenience of plotly traces later on
                # adding on ... IK itself does require to access the individual joint coordinates
                end_to_target = target - P[-1][:3, 3] # getting the last transformation [-1], to extract X, Y, Z
                err_end_to_target = sqrt_sum_aquare(end_to_target)
                error_list.append([loop, err_end_to_target])
 
                # record the angles of the best minimal error so far; yes the error can increase in further iterations
                if err_end_to_target < minimum_error:
                    minimum_error = err_end_to_target
                    least_error_angles = list_of_thetas.copy()
 
                if err_end_to_target < displacement_err:
                    solved = True
                else:
 
                    if list_of_blockers[i] != 2:
 
                        # Calculate distance between i-joint position to end effector position
                        # P[i] is position of current joint
                        # P[-1] is position of end effector
 
                        # reviewed and change code here to improve since normal vector is always Z-axis if theta is always used as rotation
                        # use the DH array matrix, because in the top-left 3x3 sub-matrix, it already contains the vectors
                        # for all 3 axis, the NORMAL axis are: top-row = X axis, middle = Y axis and last row = Z axis
 
                        # find normal of rotation plane, aka hinge axis (hinge is always normal to rotation plane)
                        normal_vector = list(P[i][2, :3])
                        plane = Plane(point=P[i][:3, 3], normal=normal_vector)
 
                        # find projection of tgt onto rotation plane
                        # https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_plane.html
                        target_point_projected = plane.project_point(target)
                        end_point_projected = plane.project_point(P[-1][:3, 3])
 
                        # find angle between projected tgt and cur_to_end
                        cur_to_end_projected = end_point_projected - P[i][:3, 3]
                        cur_to_target_projected = target_point_projected - P[i][:3, 3]
 
                        # end_target_mag = |a||b|
                        cur_to_end_projected_mag = sqrt_sum_aquare(cur_to_end_projected) # aka |a|
                        cur_to_target_projected_mag = sqrt_sum_aquare(cur_to_target_projected) # aka |b|
                        end_target_mag = cur_to_end_projected_mag * cur_to_target_projected_mag # aka |a||b|
 
                        # if the 2 vectors current-effector and current-target is already very close
                        if end_target_mag <= 0.0001:
                            cos_rot_ang = 1
                            sin_rot_ang = 0
                        else:
                            # dot product rule - https://en.wikipedia.org/wiki/Dot_product
                            # To solve for angle magnitude between 2 vectors
                            # dot product of two Euclidean vectors a and b
                            # a.b = |a||b|cos(lambda)
                            # cos_rot_ang = cos(lambda) = a.b / |a||b|
                            cos_rot_ang = (cur_to_end_projected[0] * cur_to_target_projected[0] + cur_to_end_projected[1] * cur_to_target_projected[1] + cur_to_end_projected[2] * cur_to_target_projected[2]) / end_target_mag
 
                            # cross product rule - https://en.wikipedia.org/wiki/Cross_product
                            # https://www.mathsisfun.com/algebra/vectors-cross-product.html
                            # cross product of two Euclidean vectors a and b
                            # a X b = |a||b|sin(lambda)
                            # sin_rot_ang = sin(lambda) = [a X b] / |a||b|
                            # To solve for direction of angle A->B or B->A
                            # for theta rotation (about Z axis) in right hand rule, keep using [0] and [1] for finding Z direction
                            # cross product of 3d vectors has i, j, k components
                            # after we do the projections onto the plane level, we will focus on the k component
                            sin_rot_ang = (cur_to_end_projected[0] * cur_to_target_projected[1] - cur_to_end_projected[1] * cur_to_target_projected[0]) / end_target_mag
 
                        rot_ang = math.acos(max(-1, min(1,cos_rot_ang)))
 
                        if sin_rot_ang < 0.0:
                            rot_ang = -rot_ang
 
                        # Update current joint angle values
                        list_of_thetas[i] = list_of_thetas[i] + (rot_ang * 180 / math.pi)
                        list_of_thetas[i] = (list_of_thetas[i] + 180) % 360 - 180
 
                        # clamp angle
                        if list_of_thetas[i] > angle_max[i]: list_of_thetas[i] = angle_max[i]
                        if list_of_thetas[i] < angle_min[i]: list_of_thetas[i] = angle_min[i]
 
                    elif list_of_blockers[i] == 2:
                        #list_of_thetas[i] = 90
                        # there was a bug here befoew where the blockers force it to only positive 90 deg
                        # now I have update to adopt whatever linkage data is needed e.g. -90 deg aka 270 deg
                        list_of_thetas[i] = local_linkage_data[i][1]
 
        if solved:
            break
 
    if solved == False:
        for i in range(len(list_of_thetas)):
            list_of_thetas[i] = least_error_angles[i] #return least error
            err_end_to_target = minimum_error
            P = input_linkage_angles(list_of_thetas) # forward kinematics
 
    return P, list_of_thetas, err_end_to_target, solved, loop
 
 
def add_trace(array_of_transformation_matrix):
 
    traces = []
 
    x_coordinate = []
    y_coordinate = []
    z_coordinate = []
 
    for i, transformation_matrix in enumerate(array_of_transformation_matrix):
 
        x_coordinate.append(transformation_matrix[0,3])
        y_coordinate.append(transformation_matrix[1,3])
        z_coordinate.append(transformation_matrix[2,3])
 
        if len(x_coordinate)==2:
            # optimize here comparing delta between n point and n-1 point as some of them are the same
            if x_coordinate[1]!=x_coordinate[0] or y_coordinate[1]!=y_coordinate[0] or z_coordinate[1]!=z_coordinate[0]:
 
                traces.append(go.Scatter3d(x=x_coordinate, y=y_coordinate, z=z_coordinate,
                                            #opacity=0.9,
                                            mode='lines',
 
                                            marker=dict(
                                                size=2
                                                ),
                                            line=dict(
                                                color=px.colors.qualitative.Plotly[local_linkage_data[i][0]],
                                                width=15
                                              )
                                            ))
            x_coordinate.pop(0)
            y_coordinate.pop(0)
            z_coordinate.pop(0)
 
    return traces
 
def add_axis_to_trace(traces, colour_string, axis_start, axis_end):
 
    rotate_axis = [axis_start, axis_start, axis_end]
 
    for i in range(len(colour_string)):
 
        traces.append(go.Scatter3d(x=rotate_axis[0], y=rotate_axis[1], z=rotate_axis[2],
                                   #opacity=0.7,
                                   mode='lines',
                                   marker=dict(
                                       color=colour_string[i],
                                       size=12
                                       )
                                   ))
 
        rotate_axis.append(rotate_axis.pop(0))
 
    return traces
 
def DH_array_to_hinge_list(traces, array_matrix):
 
    for j in index_of_hinges:
        transformation_matrix = array_matrix[j]
 
        # then the translations
        transient_translation = DH_matrix(0, 0, hinge_length, 0)
        positive_z_matrix = np.matmul(transformation_matrix, transient_translation)
        #transient_translation = DH_matrix(0, 0, -hinge_length, 0)
        # use this below to show the Z-axis of the coordinate frame
        # transient_translation = DH_matrix(0, 0, 0, 0)
        # negative_z_matrix = np.matmul(transformation_matrix, transient_translation)
 
 
        # grab the +/- coordinates into array
        # x/y/z = np.array([start,end])
        x_hinge = np.array([array_matrix[j][0,3],positive_z_matrix[0,3]])
        y_hinge = np.array([array_matrix[j][1,3],positive_z_matrix[1,3]])
        z_hinge = np.array([array_matrix[j][2,3],positive_z_matrix[2,3]])
 
        traces.append(go.Scatter3d(x=x_hinge, y=y_hinge, z=z_hinge,
                                   #opacity=0.7,
                                   mode='lines',
                                   marker=dict(
                                       #symbol="arrow",
                                       color="black",
                                       size=12
                                       ),
                                   line=dict(
                                    #color='purple',
                                    width=16)
                                   ))
 
 
        # # uncomment this block to show X-axis of each coordinate frame
        # # # then the translations
        # transient_translation = DH_matrix(0, 0, 0, hinge_length)
        # positive_x_matrix = np.matmul(transformation_matrix, transient_translation)
        # transient_translation = DH_matrix(0, 0, 0, -0)
        # negative_x_matrix = np.matmul(transformation_matrix, transient_translation)
 
        # # grab the +/- coordinates into array
        # # x/y/z = np.array([start,end])
        # x_hinge = np.array([negative_x_matrix[0,3],positive_x_matrix[0,3]])
        # y_hinge = np.array([negative_x_matrix[1,3],positive_x_matrix[1,3]])
        # z_hinge = np.array([negative_x_matrix[2,3],positive_x_matrix[2,3]])
 
 
        # traces.append(go.Scatter3d(x=x_hinge, y=y_hinge, z=z_hinge,
        #                             #opacity=0.7,
        #                             mode='lines',
        #                             marker=dict(
        #                                 #symbol="arrow",
        #                                 color="red",
        #                                 size=12
        #                                 ),
        #                             line=dict(
        #                             #color='purple',
        #                             width=16)
        #                             ))
 
    return(traces)
 
def append_target_to_trace(traces, target):
 
    # append target point
    traces.append(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]],
                               opacity=0.7,
                               mode='markers',
                               marker=dict(
                                   color='black',
                                   size=12
                                   )
                               ))
    return(traces)
 
if __name__ == '__main__':
 
    # target_list = [
    #     [160, 0, 160], # X, Y, Z in mm
    #     [160, 0, 160], # X, Y, Z in mm
    #     ]
 
    # rotations = np.radians(np.array([
    #     [30, 23, -16],        #start
    #     [100, 40, 20]     #end
    #     ]
    #     ))
 
    target_list = [
        [160, 0, 160], # X, Y, Z in mm
        [150, 60, 175], # X, Y, Z in mm
        ]
 
    rotations = np.radians(np.array([
        [30, 23, -16],        #start
        [100, 40, 20]     #end
        ]
        ))
 
    r = Rotation.from_euler('xyz', rotations)
 
    # Create an array of interpolation values
    num_interpolations = 20  # Number of interpolations between the two points
 
    # Interpolate between two target points
    interpolated_points = interpolate_two_points(target_list, num_interpolations)
 
    # Create the Slerp object
    slerp = Slerp([0, 1], r)
 
    # Interpolate between the two rotations
    t = np.linspace(0, 1, num_interpolations)
    interpolated_rotations = slerp(t)
 
    # Print the interpolated rotations
    for i, rot in enumerate(interpolated_rotations):
        #print(f"Step {i + 1}: {rot.as_euler('xyz', degrees=True)}")
        desired_orientation = rot.as_euler('xyz', degrees=True)
        print(f"Step {i + 1}: {desired_orientation}")
 
        target = interpolated_points[i]
        print(f"Step {i + 1}: {target}")
 
        #array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_CCD()
        #array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_Jacobian(i)
        array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_Jacobian_numerical(i)
 
        print(f"Solution: {solve_status}")
        #column_headers = ['Rows', 'X', 'Y', 'Z', 'Pos Error', 'Roll', 'Pitch', 'Yaw', 'IK Iterations', 'Solve Status', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
 
        theta_angles = [None] * 6
        Rows = i
        X = np.round(array_matrix[-1][0, 3],1)
        Y = np.round(array_matrix[-1][1, 3],1)
        Z = np.round(array_matrix[-1][2, 3],1)
        Pos_Error = err_end_to_target
        calculated_orientation = np.round(rotation_matrix_to_euler(array_matrix[-1][:3, :3]),3)
        Roll = calculated_orientation[0]
        Pitch = calculated_orientation[1]
        Yaw = calculated_orientation[2]
        IK_Iterations = iterations
        IK_time = time.time() - start_time
        solve_status = str(solve_status)
        for j, index in enumerate(index_of_angles):
            theta_angles[j] = list_of_angles[index]
        theta1 = theta_angles[0]
        theta2 = theta_angles[1]
        theta3 = theta_angles[2]
        theta4 = theta_angles[3]
        theta5 = theta_angles[4]
        theta6 = theta_angles[5]
        if i == 0:
            time_stay = 5
        else:
            time_stay = 0.03
 
        row = [
            Rows,
            X,
            Y,
            Z,
            Pos_Error,
            Roll,
            Pitch,
            Yaw,
            IK_Iterations,
            IK_time,
            solve_status,
            theta1,
            theta2,
            theta3,
            theta4,
            theta5,
            theta6,
            time_stay
            ]
 
        data.append(row)
 
        print("--- %s seconds ---" % IK_time)
 
    df = pd.DataFrame(data, columns = column_headers)
 
    df.to_csv(csv_name, index = False, header = True, mode='w+')