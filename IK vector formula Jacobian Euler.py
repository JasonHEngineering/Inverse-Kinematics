# -*- coding: utf-8 -*-
"""
@Purpose: Inverse Kinematics (vector formula Jacobian) - Euler
@author: Jason H + Engineering
More explanation @website: https://jashuang1983.wordpress.com/inverse-kinematics-robotics-jacobian/
 
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
import time
start_time = time.time()
 
pio.renderers.default='browser'
 
figure_axis_limit = 600
 
target = [200, 200, 200] # X, Y, Z in mm
desired_orientation = np.array([90,-40,30]) # roll, pitch, yaw in degree
 
displacement_err = 0.02 # acceptable error for IK in mm
orientation_err = 0.1 # acceptable error for orientation in deg
 
index_of_angles = [1, 3, 5, 8, 10, 12] #hinge for IK CCD demo in wordpress
 
config_df = pd.read_csv('config.csv')
lower_limit = list(config_df["Absolute low angle limit"])
upper_limit = list(config_df["Absolute high angle limit"])
 
index_of_hinges = index_of_angles.copy()
hinge_length = 24 # for illustrating the hinge in 3d view
max_iter = 2000 # iterations allow for IK CCD
step_size=0.01 #step size for jacobian delta_theta
error_list = []
 
local_linkage_data = [
    [0,0,0,0,0], #0
    [1,0,0,0,0], # for theta - yaw on the first hinge     #1
    [1,0,0,0,100], # L100 on first                          #2
    [2,0,0,0,0], # for theta - yaw on the second hinge     #3
    [2,0,0,0,100], # L100 on second hinge                   #4
    [3,0,0,0,0], # for theta - yaw on the third hinge     #5
    [3,0,0,0,100], # L100 on second hinge                   #6
    [4,0,-90,0,0], # convert hinge from yaw to pitch        #7
    [4,0,0,0,0], # for theta - pitch on the fourth          #8
    [4,0,0,0,100], # L100 on third hinge                    #9
    [5,0,0,0,0], # for theta - pitch on the fifth           #10
    [5,0,0,0,100], # L100 on fourth hinge                   #11
    [6,0,0,0,0], # for theta - pitch on the sixth           #12
    [6,0,0,0,100] # L100 on fourth hinge                    #13
    ]
 
global array_matrix
array_matrix = np.array([None] * len(local_linkage_data)) # global declaration from V24 onwards to save run time
 
list_of_thetas = [0] * len(local_linkage_data)
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
 
    return_matrix = [None] * len(local_linkage_data)
 
    for i, linkage in enumerate(local_linkage_data):
 
        try:
            if (array_matrix[i] == None):
                boolean_state = (array_matrix[i] == None)
        except Exception:
            boolean_state = ((array_matrix[i] == None).any())
 
        if boolean_state:
            array_matrix[i] = DH_matrix(linkage[1], linkage[2], linkage[3], linkage[4])
 
        elif i in index_of_angles:
            array_matrix[i] = DH_matrix(linkage[1], linkage[2], linkage[3], linkage[4])
 
    for j, matrix in enumerate(array_matrix):
        if j == 0:
            return_matrix[j] = matrix
 
        if j!=0:
            return_matrix[j] = np.matmul(return_matrix[j-1], matrix)
 
    return(return_matrix)
 
def sqrt_sum_aquare(input_list):
    sum_square = 0
    for value in input_list:
        sum_square += value*value
    return(math.sqrt(sum_square))
 
def Inverse_Kinematics_Jacobian():
 
    solved = False
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
        # print(current_orientation)
        error_orientation = desired_orientation - current_orientation            
        error_list.append([loop, err_end_to_target])
 
        # record the angles of the best minimal error so far; yes the error can increase in further iterations
        if err_end_to_target < minimum_error:
            minimum_error = err_end_to_target
            least_error_angles = list_of_thetas.copy()
 
        if (err_end_to_target < displacement_err) and (error_orientation < orientation_err).all():
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
                if list_of_thetas[value] > angle_max[value]: list_of_thetas[value] = angle_max[value]
                if list_of_thetas[value] < angle_min[value]: list_of_thetas[value] = angle_min[value]               
 
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
 
 
    #array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_CCD()
    array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_Jacobian()
     
    print("--- %s seconds ---" % (time.time() - start_time))
     
    # add X/Y/Z axis
    axis_start = np.array([0,0])
    axis_end = np.array([-figure_axis_limit,figure_axis_limit])
    colour_list = ['rgb(0,100,80)', 'rgb(0,176,246)','rgb(231,107,243)']
 
    traces = add_trace(array_matrix)
    traces = add_axis_to_trace(traces, colour_list, axis_start, axis_end) # add the 3 axis X, Y, Z
    traces = DH_array_to_hinge_list(traces, array_matrix) # add hinges
    traces = append_target_to_trace(traces, target) # add target point
 
    fig = go.Figure(data=traces)
 
    annotation_text = (
        "Target Coordinates: " + str(target) +
        "<br>" + "Actual Coordinates: " + str(np.round(array_matrix[-1][:3, 3],1)) +
        "<br>" + "Error (mm): " + str(round(err_end_to_target,3)) +
        "<br>" + "IK Iterations: " + str(iterations+1) + " (max: " + str(max_iter) + ")" +
        "<br>"
        )
 
    annotation_angle = ""
    for i,index in enumerate(index_of_angles):
        annotation_angle = annotation_angle + "Axis " + str(i+1) + ": " + str(round(list_of_angles[index],1)) + "°" + "<br>"
 
    annotation_text = annotation_text + annotation_angle
 
    fig.add_annotation(text=annotation_text,
                  xref="paper", yref="paper",
                  x=1, y=0.9,
                  align='left',
                  showarrow=False)
 
    fig.update_layout(showlegend=False,
                        #height=800,
                        #width=1200
                        #font=dict(size=8)
                        margin=dict(
                            b=0, #bottom margin 0px
                            l=0, #left margin 0px
                            r=0, #right margin 0px
                            t=0, #top margin 0px
                        ),
                        scene = dict(
                            xaxis = dict(range=[-figure_axis_limit,figure_axis_limit],),
                            yaxis = dict(range=[-figure_axis_limit,figure_axis_limit],),
                            zaxis = dict(range=[-figure_axis_limit,figure_axis_limit],),),
                        )
    fig.show()
 
    # plot error_list on another figure
    error_array = np.array(error_list)
    error_df = pd.DataFrame(error_list, columns = ["Iterations","Error"])
    error_df["Iterations"]=error_df["Iterations"]+1
    error_df['Iterations'] = error_df['Iterations'].astype(str) # convert continuous colours to discrete
    fig2 = px.line(error_df, y="Error",color="Iterations", markers=True)
    fig2.show()
 
    print("--- %s seconds ---" % (time.time() - start_time))