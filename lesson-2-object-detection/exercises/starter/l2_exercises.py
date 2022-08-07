# ---------------------------------------------------------------------
# Exercises from lesson 2 (object detection)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib

import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

precision_all =[]
recall_all =[]

# Exercise C2-4-6 : Plotting the precision-recall curve
def plot_precision_recall(): 
    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    # by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.
    # Please create a 2d scatter plot of all precision/recall pairs 
    print("precision_all - ", precision_all)
    print("recall_all - ", recall_all)
    
    plt.scatter(x=recall_all, y=precision_all)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    return

# Exercise C2-3-4 : Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):

    if len(det_performance_all)==0 :
        print("no detections for conf_thresh = " + str(conf_thresh))
        return
    
    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]
    #print("det_performance_all -", det_performance_all)

    pos_negs = []

    for item in det_performance_all:
        #print("item :",item)
        pos_negs.append(item[2])
    
    pos_negs_arr = np.asarray(pos_negs) 
    print("/n pos_negs_arr-",pos_negs_arr)
    positives = sum(pos_negs_arr[:,0])
    true_positives = sum(pos_negs_arr[:,1])
    false_positives = sum(pos_negs_arr[:,2])
    false_negatives = sum(pos_negs_arr[:,3])

    print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    # compute precision
    precision = true_positives / (true_positives + false_positives)
    
    # compute recall 
    recall = true_positives/ ( true_positives + false_negatives )

    precision_all.append(precision)
    recall_all.append(recall)    

    print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")    
    



# Exercise C2-3-2 : Transform metric point coordinates to BEV space
def pcl_to_bev(lidar_pcl, configs, vis=True):

    #print("configs: \n", configs)
    # compute bev-map discretization by dividing x-range by the bev-image height
    discretization = (configs.lim_x[1] - configs.lim_x[0])/configs.bev_height

    #print("discretization : ", discretization)
    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:5,])
    lidar_pcl_cpy[:,0] = np.int_(lidar_pcl_cpy[:,0] / discretization)
    #print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    #print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:5,])


    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    centre_adjustment = (configs.bev_width)/2
    lidar_pcl_cpy[:,1] = np.int_((lidar_pcl_cpy[:,1])/discretization + centre_adjustment)
    #print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy [:,2] = lidar_pcl_cpy [:,2] - configs.lim_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    sort = np.lexsort((-lidar_pcl_cpy[:,2],lidar_pcl_cpy[:,1],lidar_pcl_cpy[:,0]))
    print("sort: \n", sort[:10,])

    lidar_pcl_cpy = (lidar_pcl_cpy[sort])
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _ , uniqueR = np.unique(lidar_pcl_cpy[:,0:2], axis=0, return_index=True)
    print("uniqueR: \n", len(uniqueR))

    lidar_pcl_cpy = (lidar_pcl_cpy[uniqueR])
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # assign the height value of each unique entry in lidar_top_pcl to the height map and 
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    normalize_height = configs.lim_z[1]-configs.lim_z[0]
    lidar_pcl_cpy [:,2] = np.int_((lidar_pcl_cpy [:,2]/normalize_height*255)) #)) #
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy.max : ", np.max(lidar_pcl_cpy[:,2]))
    print("lidar_pcl_cpy.min : ",np.min(lidar_pcl_cpy[:,2]))
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    print("lidar_pcl_cpy before intensity assignment: \n",lidar_pcl_cpy [lidar_pcl_cpy[:,3]>1.0,3])

    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0


    print("lidar_pcl_cpy after intensity assignment: \n",lidar_pcl_cpy [lidar_pcl_cpy[:,3]>1.0,3])

    print("lidar_pcl_cpy after intensity assignment: \n", lidar_pcl_cpy[:10,])

    sort_int = np.lexsort((-lidar_pcl_cpy[:,3],lidar_pcl_cpy[:,1],lidar_pcl_cpy[:,0]))
    print("sort_int: \n", sort_int[:10,])

    lidar_pcl_cpy = (lidar_pcl_cpy[sort_int])
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # only keep one point per grid cell
    _, unique_int = np.unique(lidar_pcl_cpy[:,0:2], axis=0, return_index=True)
    print("unique_int: \n", len(unique_int))

    lidar_pcl_cpy = (lidar_pcl_cpy[unique_int])
    print("lidar_pcl_cpy.shape : ",lidar_pcl_cpy.shape)
    print("lidar_pcl_cpy: \n", lidar_pcl_cpy[:10,])

    # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_cpy[2609:, 0]), np.int_(lidar_pcl_cpy[2609:, 1])] = lidar_pcl_cpy[2609:, 3] / (np.amax(lidar_pcl_cpy[:, 3])-np.amin(lidar_pcl_cpy[:, 3]))
    print("intensity_map.shape : ",intensity_map.shape)
    print("intensity_map: \n", intensity_map[:10,])

    # visualize intensity map
    if vis:
       img_intensity = intensity_map * 256
       img_intensity = img_intensity.astype(np.uint8)
       while (1):
           cv2.imshow('img_intensity', img_intensity)
           if cv2.waitKey(10) & 0xFF == 27:
               break
       cv2.destroyAllWindows()
    return