# -*- coding: utf-8 -*-
"""
@Purpose: Inverse Kinematics (CCD) – Cyclic Coordinate Descent in 3d space
@author: Jason H + Engineering
More explanation @website: https://jashuang1983.wordpress.com/inverse-kinematics-cyclic-coordinate-descent-in-3d-space/

"""

import pandas as pd
import random
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from skspatial.objects import Plane, Point, Vector
import math
import plotly.io as pio
 
pio.renderers.default='browser'
 
figure_axis_limit = 300
list_of_thetas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
list_of_rotated_angle = list_of_thetas
angle_max = [0, 120, 0, 120, 0, 120, 0, 0, 120, 0, 120, 0, 120, 0]
angle_min = [0, -120, 0, -120, 0, -120, 0, 0, -120, 0, -120, 0, -120, 0]
list_of_blockers = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0] 
index_of_angles = [i for i, x in enumerate(list_of_blockers) if x == 1]
index_of_hinges = [x-1 for x in index_of_angles] # hinges are currently assumed before each rows with theta entry
hinge_length = 12 # for illustrating the hinge in 3d view
max_iter = 1000 # iterations allow for IK CCD
error_list = []
 
# [link number, theta about Z, alpha about X, d in Z direction, a or r in X direction]
# Link Length Parameter
# Note: Although DH allows theta and alpha, convert all angles to theta only for code simplicity
local_linkage_data = [
    [0,0,0,0,0],
    [1,0,0,0,0], # for theta - yaw on the first hinge     
    [1,0,0,0,100], # L100 on first hinge    
    [2,0,0,0,0], # for theta - yaw on the second hinge     
    [2,0,0,0,100], # L100 on second hinge  
    [3,0,0,0,0], # for theta - yaw on the third hinge     
    [3,0,0,0,100], # L100 on second hinge  
    [4,0,-90,0,0], # convert hinge from yaw to pitch 
    [4,0,0,0,0], # for theta - pitch on the fourth hinge
    [4,0,0,0,100], # L100 on third hinge  
    [5,0,0,0,0], # for theta - pitch on the fifth hinge
    [5,0,0,0,100], # L100 on fourth hinge  
    [6,0,0,0,0], # for theta - pitch on the sixth hinge
    [6,0,0,0,100] # L100 on fourth hinge  
    ]
 
 
def DH_matrix(theta, alpha, delta, rho):
 
    transient_matrix = np.eye(4)
     
    # Handle 3d DH parameters, row-by-row, left-to-right
    transient_matrix[0,0]=np.cos(theta/180*np.pi)
    transient_matrix[0,1]=-np.sin(theta/180*np.pi)    
    transient_matrix[0,2]=0   
    transient_matrix[0,3]=rho    
 
    transient_matrix[1,0]=np.sin(theta/180*np.pi)*np.cos(alpha/180*np.pi)
    transient_matrix[1,1]=np.cos(theta/180*np.pi)*np.cos(alpha/180*np.pi)  
    transient_matrix[1,2]=-np.sin(alpha/180*np.pi)     
    transient_matrix[1,3]=-np.sin(alpha/180*np.pi) * delta    
 
    transient_matrix[2,0]=np.sin(theta/180*np.pi)*np.sin(alpha/180*np.pi)
    transient_matrix[2,1]=np.cos(theta/180*np.pi)*np.sin(alpha/180*np.pi)  
    transient_matrix[2,2]=np.cos(alpha/180*np.pi)     
    transient_matrix[2,3]=np.cos(alpha/180*np.pi) * delta   
     
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
 
    return(array_matrix)
 
def sqrt_sum_aquare(input_list):
    sum_square = 0
    for value in input_list:
        sum_square += value*value
    return(math.sqrt(sum_square))
 
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
 
    return c
 
def Inverse_Kinematics_CCD(target):
    solved = False
    err_end_to_target = math.inf
    minimum_error = math.inf
    err_min = 0.1
     
    for loop in range(max_iter):
        for i in range(len(local_linkage_data)-1, -1, -1):
             
            if list_of_blockers[i] != 0:
 
                P = input_linkage_angles(list_of_thetas) # forward kinematics 
                end_to_target = target - P[-1][:3, 3]
                err_end_to_target = sqrt_sum_aquare(end_to_target)
                error_list.append([loop, err_end_to_target])
                 
                # record the angles of the best minimal error so far; yes the error can increase in further iterations
                if err_end_to_target < minimum_error:
                    minimum_error = err_end_to_target
                    least_error_angles = list_of_thetas.copy()                  
                 
                if err_end_to_target < err_min:
                    solved = True
                else:
                    # Calculate distance between i-joint position to end effector position
                    # P[i] is position of current joint
                    # P[-1] is position of end effector
 
                    # find normal of rotation plane, aka hinge axis (hinge is always normal to rotation plane)
                    # a first vector on plane
                    vector_1 = P[i+1][:3, 3] - P[i][:3, 3]
                    # a second vector by simulating a +ve 90 deg rotation on hinge
                    list_of_rotated_angle = list_of_thetas.copy() # list have to copy(), else it becomes only a reference, modify one will modify both
                    list_of_rotated_angle[i] = list_of_rotated_angle[i]+90
 
                    Q = input_linkage_angles(list_of_rotated_angle) # forward kinematics 
                    vector_2 = Q[i+1][:3, 3] - Q[i][:3, 3] # Q[i] should be = to P [i]
 
                    # normal vector to the rotation plane is cross product of vector 1 and vector 2            
                    normal_vector = cross(vector_1, vector_2)
 
                    # find projection of tgt onto rotation plane
                    # https://scikit-spatial.readthedocs.io/en/stable/gallery/projection/plot_point_plane.html
                    plane = Plane(point=P[i][:3, 3], normal=normal_vector)
                    target_point_projected = plane.project_point(target)
 
                    end_point_projected = plane.project_point(P[-1][:3, 3])
                     
                    # find angle between projected tgt and cur_to_end
                    cur_to_end_projected = end_point_projected - P[i][:3, 3]
                    cur_to_target_projected = target_point_projected - P[i][:3, 3]
 
                    # end_target_mag = |a||b|    
                    cur_to_end_projected_mag = sqrt_sum_aquare(cur_to_end_projected)
                    cur_to_target_projected_mag = sqrt_sum_aquare(cur_to_target_projected)
                    end_target_mag = cur_to_end_projected_mag * cur_to_target_projected_mag
 
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
        transient_translation = DH_matrix(0, 0, -hinge_length, 0)
        negative_z_matrix = np.matmul(transformation_matrix, transient_translation)
         
        # grab the +/- coordinates into array
        # x/y/z = np.array([start,end])
        x_hinge = np.array([negative_z_matrix[0,3],positive_z_matrix[0,3]])
        y_hinge = np.array([negative_z_matrix[1,3],positive_z_matrix[1,3]])
        z_hinge = np.array([negative_z_matrix[2,3],positive_z_matrix[2,3]])
         
 
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
     
    target = [random.randint(-200, 200), random.randint(-200, 200), random.randint(-200, 200)]
     
    array_matrix, list_of_angles, err_end_to_target, solve_status, iterations = Inverse_Kinematics_CCD(target)
     
    # add X/Y/Z axis
    axis_start = np.array([0,0])
    axis_end = np.array([-figure_axis_limit,figure_axis_limit])
    colour_list = ['rgb(0,100,80)', 'rgb(0,176,246)','rgb(231,107,243)']
 
    traces = add_trace(array_matrix)    
    traces = add_axis_to_trace(traces, colour_list, axis_start, axis_end) # add the 3 axis 
    traces = DH_array_to_hinge_list(traces, array_matrix) # add hinges
    traces = append_target_to_trace(traces, target) # add target point
 
    fig = go.Figure(data=traces)
 
    annotation_text = (
        "Target Coordinates: " + str(target) +
        "<br>" + "Actual Coordinates: " + str(np.round(array_matrix[-1][:3, 3],1)) +
        "<br>" + "Error (mm): " + str(round(err_end_to_target,3)) +
        "<br>" + "IK CCD Iterations: " + str(iterations+1) + " (max: " + str(max_iter) + ")" +
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