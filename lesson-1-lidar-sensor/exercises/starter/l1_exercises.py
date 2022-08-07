# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
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
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

# extract range image from frame
    print("Exercise C1-5-5")
    # load range image
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    print(" Range Image Shape : ", ri.shape )
    
    print('max. Intensity  = ' + str(round(np.amax(ri[:,:,1]),2)) + 'm')
    print('min. Intensity = ' + str(round(np.amin(ri[:,:,1]),2)) + 'm')


    # extract range data and convert to 8 bit
    ri[ri<0]=0.0
    
    print('max. Intensity  = ' + str(round(np.amax(ri[:,:,1]),2)) + 'm')
    print('min. Intensity = ' + str(round(np.amin(ri[:,:,1]),2)) + 'm')

    # map value range to 8bit
    ri_range = ri[:,:,1]
    ri_range = np.amax(ri_range)/2 * ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))

    print(" ri_range Shape : ", ri_range.shape )

    print('max. Intensity  = ' + str(round(np.amax(ri_range[:,:]),2)) + 'm')
    print('min. Intensity = ' + str(round(np.amin(ri_range[:,:]),2)) + 'm')

    img_range = ri_range.astype(np.uint8)
    

    # focus on +/- 45° around the image center
    deg45 = int(img_range.shape[1] / 8)
    ri_center = int(img_range.shape[1]/2)
    img_range = img_range[:,ri_center-deg45:ri_center+deg45]

    print('max. val = ' + str(round(np.amax(img_range[:,:]),2)))
    print('min. val = ' + str(round(np.amin(img_range[:,:]),2)))

    cv2.imshow('intensity_image', img_range)
    cv2.waitKey(10000)
    


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
        
    # extract range data and convert to 8 bit
    ri_range = ri[:,:,0]
    ri_range = ri_range * 256 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    
    print(" Range Image Shape : ", ri.shape )
    # visualize range image
    #cv2.imshow('range_image', img_range)
    #cv2.waitKey(50000)

    # compute vertical field-of-view from lidar calibration 
    # get lidar calibration data
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]

    #print("Stats : ", frame.context.stats)
    #print("calib_lidar : ", calib_lidar)

    # compute vertical field of view (vfov) in rad
    vfov_rad = calib_lidar.beam_inclination_max - calib_lidar.beam_inclination_min
    print("vfov_rad : ", vfov_rad)

    # compute pitch resolution and convert it to angular minutes
    ri[ri<0]=0.0

    print('max. range = ' + str(round(np.amax(ri[:,:,0]),2)) + 'm')
    print('min. range = ' + str(round(np.amin(ri[:,:,0]),2)) + 'm')

    pitch_res_rad = vfov_rad / ri.shape[0]
    print("pitch_res_rad : ", pitch_res_rad)
    pitch_res_deg = pitch_res_rad * 180 / np.pi
    print("pitch angle resolution = " + '{0:.2f}'.format(pitch_res_deg) + "°")
    print("pitch angular = " + '{0:.2f}'.format(pitch_res_deg*60) + "'")




# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")    
    
    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = 0

    # loop over all labels
    for label in frame.laser_labels:
        #print(type(label))
        #print("label.type:",label.type)
        #print("label.TYPE_VEHICLE:",label.TYPE_VEHICLE)
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            num_vehicles += 1
            
    print("number of labeled vehicles in current frame = " + str(num_vehicles))