import cv2
import numpy as np
import open3d as o3d
import laspy
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3d
import os

#length, width, height
car_anchor = []
pedestrian_anchor = []
cyclist_anchor = []

def get_anchor_data(object_ground_truthPath):
    object_ground_truth_lines = []
    with open(object_ground_truthPath) as object_ground_truthFile:
        object_ground_truth_lines = object_ground_truthFile.readlines()

    kitti_data = []

    objects = ['Coupe', 'Truck', 'SUV', 'Bus', 'Hatchback', 'Cyclist', 'Sedan', 'Van', 'Motorcyclist', 'PickupTruck', 'Pedestrian']
    cars = ['Coupe', 'SUV', 'Hatchback', 'Sedan', 'PickupTruck']


    for line in object_ground_truth_lines:
        obj_category = line.split()[2]
        if not (obj_category in objects):
            continue
        
        #height, length, width
        anchor = [line.split()[10], line.split()[12], line.split()[11]]
        if obj_category in cars:
            #car
            car_anchor.append(anchor)
        elif obj_category == 'Pedestrian':
            #Pedestrian
            pedestrian_anchor.append(anchor)
        elif obj_category == 'Cyclist':
            #Cyclist
            cyclist_anchor.append(anchor)


def get_anchors(path, filename):
    object_ground_truthPath = path.replace("Depth", "Object_GroundTruth")
    object_ground_truthPath = object_ground_truthPath.replace("png", "txt")

    depthImage = cv2.imread(path)

    blueChannel = depthImage[:,:,0]
    greenChannel = depthImage[:,:,1]
    redChannel = depthImage[:,:,2]

    get_anchor_data(object_ground_truthPath)
    

def is_in_skip(subdir):
    dir_skips = ['DEGRADATION', 'SEVERE_DEGRADATION', 'Without_Pedestrian', 'Without_TrafficBarrier']
    dirs = ['NO_DEGRADATION', 'With_Pedestrian', 'With_TrafficBarrier']
    for di in dirs:
        if di not in subdir:
            for d in dir_skips:
                if d in subdir:
                    return True
    return False


def main():
    directory = '/path/to/dir/Depth'
    file_counter = 0
    
    for subdir, dirs, files in os.walk(directory):
        if is_in_skip(subdir):
            #print(subdir)
            continue

        #print(subdir)
        #print(dirs)
        #continue
        for file in files:
            if file == None:
                continue
            filename = str(file_counter).zfill(6)
            path = os.path.join(subdir, file)
            path = path.replace("\\", "/")
            get_anchors(path, filename)



    car_anchors = np.array(car_anchor).astype(np.float)      
    car_height = np.mean(car_anchors[:,0])
    car_length = np.mean(car_anchors[:,1])
    car_width = np.mean(car_anchors[:,2])
    print("Car: height: {}, length: {}, width: {}".format(car_height,car_length,car_width))
    pedestrian_anchors = np.array(pedestrian_anchor).astype(np.float)
    pedestrian_height = np.mean(pedestrian_anchors[:,0])
    pedestrian_length = np.mean(pedestrian_anchors[:,1])
    pedestrian_width = np.mean(pedestrian_anchors[:,2])
    print("pedestrian: height: {}, length: {}, width: {}".format(pedestrian_height,pedestrian_length,pedestrian_width))
    cyclist_anchors = np.array(cyclist_anchor).astype(np.float)
    cyclist_height = np.mean(cyclist_anchors[:,0])
    cyclist_length = np.mean(cyclist_anchors[:,1])
    cyclist_width = np.mean(cyclist_anchors[:,2])
    print("cyclist: height: {}, length: {}, width: {}".format(cyclist_height,cyclist_length,cyclist_width))
    

main()



