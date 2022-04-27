import cv2
import numpy as np
import open3d as o3d
import laspy
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3d
import os
from pathlib import Path

def norm(number):
    return (number - 0.0)/(255.0 - 0.0)

#generate ground truth data in KITTI format
def get_kitti_data(object_ground_truthPath):
    #read lines in ground truth file into list
    object_ground_truth_lines = []
    with open(object_ground_truthPath) as object_ground_truthFile:
        object_ground_truth_lines = object_ground_truthFile.readlines()

    kitti_data = []
    #maximum values defined by point_cloud_size in config file
    max_x = 33.0
    max_y = 3.0
    max_z = 70.00
    min_x = -33.0
    min_y = -1.0
    min_z = 0.00

    #our y val -> if our y val = -1.6 -> should be 0 -> everything + 1.6 ?

    #maybe generalize to Car, Cyclist, Pedestrian
    #or generalize to those...
    #Database Pedestrian: 2207
    #Database Car: 14357
    #Database Cyclist: 734
    #Database Van: 1297
    #Database Truck: 488
    #Database Tram: 224
    #Database Misc: 337
    #Database Person_sitting: 56
    objects = ['Coupe', 'Truck', 'SUV', 'Bus', 'Hatchback', 'Cyclist', 'Sedan', 'Van', 'Motorcyclist', 'PickupTruck', 'Pedestrian']
    cars = ['Coupe', 'SUV', 'Hatchback', 'Sedan', 'PickupTruck']

    for line in object_ground_truth_lines:
        obj_category = line.split()[2]
        if not (obj_category in objects):
            continue
        if obj_category in cars:
            obj_category = 'Car'
        x = float(line.split()[13])
        y = float(line.split()[14])
        z = float(line.split()[15])

        if x > max_x or y > max_y or z > max_z:
            continue
        if x < min_x or y < min_y or z < min_z:
            continue
        
        #height, length, width opposed to normal height, width, length
        #y value of obj center + obj_height/2 to make kitti format with x,y,z being the bottom center of obj bounding box
        append_line = [obj_category, line.split()[3], line.split()[4], line.split()[5], line.split()[6], line.split()[7], line.split()[8], line.split()[9],  line.split()[10], line.split()[12], line.split()[11], line.split()[13], str(float(line.split()[14]) + float(line.split()[10]) / 2), line.split()[15], line.split()[16]]
        kitti_data.append(append_line)

    if not kitti_data:
        return False
    
    return kitti_data

#function to generate the point cloud from a depth image
def gen_pc(path, filename):
    #get object ground truth path from depth image path ('path') -> Object_GroundTruth folder needs to be
    #positioned in same folder as Depth folder
    object_ground_truthPath = path.replace("Depth", "Object_GroundTruth")
    object_ground_truthPath = object_ground_truthPath.replace("png", "txt")

    depthImage = cv2.imread(path)
    #split depth image into seperate channels
    blueChannel = depthImage[:,:,0]
    greenChannel = depthImage[:,:,1]
    redChannel = depthImage[:,:,2]
    
    
    #--------- CONSTANTS -------------
    #focal distance (distance from camera center to image plane)
    f = 2015.0
    #coordinates of principal point on image
    px = 960.0
    py = 540.0

    max_distance = 70.0 #max distance for objects in KITTI dataset

    #KITTI fov
    fov_v = 26.8
    fov_h = 60
    resolution = 64
    vert_res = np.radians(fov_v/resolution) * 1000

    lidar_hor_mrad = 1.570796 #0.09 deg - velodyne hdl-64e lidar
    lidar_ver_mrad = vert_res #26.8/64 = ~0.42 deg - velodyne hdl-64e lidar

    #----------------------------------
    
    #get image width/height from depth image
    image_width = depthImage.shape[1]
    image_height = depthImage.shape[0]

    #calculate horizontal/vertical resolution of lidar in degrees (mrad to deg)
    lidar_hor_deg = math.degrees(lidar_hor_mrad/1000)
    lidar_ver_deg = math.degrees(lidar_ver_mrad/1000)

    #generate matrix of x-values with size of image 
    #e.g. for image with width = 1920 and height = 1080
    #resulting matrix shape = (1080,1920)
    #each *row* holds the same values e.g. (0, 1, 2, ...,1918,1919) 
    x_coords = np.arange(image_width)
    x_coords = np.repeat(x_coords[np.newaxis,:], image_height, 0)
    
    #generate matrix of y-values with size of image
    #e.g. for image with width = 1920 and height = 1080
    #resulting matrix shape = (1080,1920)
    #each *column* holds the same values e.g. (0, 1, 2, ...,1078,1079)
    y_coords = np.arange(image_height)
    y_coords = np.repeat(y_coords[np.newaxis,:], image_width, 0).transpose()
    
    #generate arrays holding all degrees of the lidar rays between -fov/2 and fov/2
    x_coords_lidar = np.arange(-(fov_h/2),(fov_h/2), lidar_hor_deg)
    y_coords_lidar = np.arange(-(fov_v/2),(fov_v/2), lidar_ver_deg)
    
    x_size = x_coords_lidar.size
    y_size = y_coords_lidar.size
    
    #generate matrices for vertical/horizontal degrees from previously generated arrays
    x_coords_lidar = np.repeat(x_coords_lidar[np.newaxis,:], y_size, 0)
    y_coords_lidar = np.repeat(y_coords_lidar[np.newaxis,:], x_size, 0).transpose()

    #boolean matrix with same size as depth image -> used to mark pixels which get intersected by LiDAR ray
    bool_mat = np.full((image_height,image_width), False)
    
    
    #--- Spherical Coordinate representation start ---    
    
    #vector should be unit vector, therefore radius/length = 1
    r = 1
    
    #the axes of the Apollo Synthetic dataset differ from the standard axes of spherical coordinates
    #-> y and z need to be interchanged
    
    #x=x
    #y=z
    #z=y

    x = r * np.sin(np.deg2rad(90 + y_coords_lidar)) * np.cos(np.deg2rad(90 + x_coords_lidar))
    z = r * np.sin(np.deg2rad(90 + y_coords_lidar)) * np.sin(np.deg2rad(90 + x_coords_lidar))
    y = r * np.cos(np.deg2rad(90 + y_coords_lidar))

    #--- Spherical Coordinate representation end ---


    #--- line-plane intersection start ---
    
    #n = normal vector to the plane -> (0,0,1) = normal vector pointing towards principal point on image
    n = np.array((0,0,1))
    #p0 = point on the plane -> (0,0,f) = vector pointing towards principal point on image
    #as the distance to the image is the focal lenght f, p0 lies on the plane
    p0 = np.array((0,0,f))
    #p0 = point on the line -> (0,0,0) = origin of ray, therefore guaranteed to be on the line (ray)
    l0 = np.array((0,0,0))
    
    #l = vector in the direction of the simulated ray
    l = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1), y.reshape(y.shape[0] * y.shape[1], 1), z.reshape(z.shape[0] * z.shape[1], 1)))
    
    p0l0 = p0 - l0
    dot_p0_n = np.dot(p0l0, n)
    ln = np.dot(l, n)

    d = dot_p0_n/ln
    d = d[:,np.newaxis]

    #point on image plane, where intersection takes place
    p = l * d

    #add principal point offset
    y_ = p[:,1] + py
    x_ = p[:,0] + px

    y_ = y_.round().astype(int)
    x_ = x_.round().astype(int)

    #exclude points which lie outside of the image size
    indices = np.where((x_>=0)*(x_<image_width))
    x_ = np.take(x_, indices)
    y_ = np.take(y_, indices)

    indices = np.where((y_>=0)*(y_<image_height))
    x_ = np.take(x_, indices[1])
    y_ = np.take(y_, indices[1])

    #set hit points to True in boolean matrix
    bool_mat[y_, x_] = True
    
    #--- line-plane intersection end ---
    

    #--- 3D coordinate calculation ---
    
    red = np.array(redChannel)
    green = np.array(greenChannel)

    #generate depth value from red and green channel of depth image -> equation given
    # at https://apollo.auto/synthetic.html
    Z_mat = (norm(red) + norm(green)/255.0) * 655.36

    #calculate X and Y
    X_mat = ((x_coords - px) * Z_mat)/f
    Y_mat = ((y_coords - py) * Z_mat)/f

    #discard points which are not hit by the LiDAR
    X_mat = X_mat[bool_mat]
    Y_mat = Y_mat[bool_mat]
    Z_mat = Z_mat[bool_mat]

    #discard points which are further away than the defined maximum distance
    z_indices = np.where(Z_mat <= max_distance)
    X_mat = X_mat[z_indices]
    Y_mat = Y_mat[z_indices]
    Z_mat = Z_mat[z_indices]

    #combined X,Y and Z matrices to one matrix representing the point cloud
    pointsarr = np.hstack((X_mat.reshape(X_mat.shape[0], 1), Y_mat.reshape(Y_mat.shape[0], 1), Z_mat.reshape(Z_mat.shape[0], 1)))

    #generate label files corresponding to the generated point cloud
    data = get_kitti_data(object_ground_truthPath)
    if data == False:
        return False
    #pc_with_bounding_boxes(object_ground_truthPath, pointsarr, colarr_o3d)
    #make_o3d_pointcloud(pointsarr, colarr_o3d, filename)
    #make_lasfile(pointsarr, colarr_las, filename)

    #openpcdet -> transform coordinates 

    #x -> points towards front
    #y -> points towards left
    #z -> points towards top

    #for us that means switching:
    #our x is pcdet y
    #our y is pcdet z
    #our z is pcdet x

    #maybe our x has to be flipped to fit pcdet y
    #my_pc.astype(np.float32).tofile("cloud.bin")

    #version 1 -> y flipped because our y from bottom up -> pcdet z from top down
    zeros = np.zeros(Z_mat.shape[0])
    Y_flipped = (Y_mat * (-1))# + 1.0
    X_flipped = X_mat * (-1)
    #Y_flipped = Y_mat
    #X_flipped = X_mat
    pointsarr_v1 = np.hstack((Z_mat.reshape(Z_mat.shape[0], 1), X_flipped.reshape(X_flipped.shape[0], 1), Y_flipped.reshape(Y_flipped.shape[0], 1), zeros.reshape(zeros.shape[0], 1)))
    
    pointsarr_v1.astype(np.float32).tofile("dataset/pc/" + filename + ".bin")
  

    with open("dataset/label/" + filename + ".txt", "w") as output:
        for line in data:
            space = " "
            split_line = space.join(line) 
            output.write(split_line + "\n")


    cv2.imwrite("dataset/depthImg/" + filename + ".png", depthImage)

    #pointsarr_v2.astype(np.float32).tofile("cloud_v2_.bin")
    
    #cv2.imwrite("dataset/rgb/" + filename + ".jpg", rgbImage)

    return True

#function to check if directory needs to be skipped, as we only want frames depth images which
#have 'No_DEGRADATION' and are 'With_Pedestrian' and 'With_TrafficBarrier'
#therefore directories which do not fit this requirement need to be skipped
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
    #path to 'Depth' directory of Apollo Synthetic dataset
    directory = '/path/to/dir/Depth'
    #counter to generate filenames according to the KITTI naming format (6-digit counter with leading 0's)
    file_counter = 0
    
    #create 'dataset' directory with subdirectories 'pc', 'label' and 'depthImg'
    Path("dataset/pc").mkdir(parents=True, exist_ok=True)
    Path("dataset/label").mkdir(parents=True, exist_ok=True)
    Path("dataset/depthImg").mkdir(parents=True, exist_ok=True)
    
    #iterate through all subdirectories and files of 'Depth' directory
    for subdir, dirs, files in os.walk(directory):
        #skip unwanted directories
        if is_in_skip(subdir):
            continue

        #iterate through every file in subdirecory i. e. every depth image in the subdirectory
        for file in files:
            if file == None:
                continue
            #generate filename that fits the 6-digit KITTI naming format (6-digit counter with leading 0's)
            filename = str(file_counter).zfill(6)
            path = os.path.join(subdir, file)
            path = path.replace("\\", "/")
            #generate point cloud from depth image
            ret = gen_pc(path, filename)
            #if successful point cloud generation increase file counter
            if ret:
                file_counter += 1
                 

main()



