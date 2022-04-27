import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3d
import mayavi.mlab as mlab

def norm(number):
    return (number - 0.0)/(255.0 - 0.0)


def main():

    depthPath = '/path/to/depth_image/image_name.png'
    
    object_ground_truthPath = depthPath.replace("Depth", "Object_GroundTruth")
    object_ground_truthPath = object_ground_truthPath.replace("png", "txt")

    depthImage = cv2.imread(depthPath)

    blueChannel = depthImage[:,:,0]
    greenChannel = depthImage[:,:,1]
    redChannel = depthImage[:,:,2]
    
    #--------- CONSTANTS -------------
    f = 2015.0
    px = 960.0
    py = 540.0

    max_distance = 120.0

    #KITTI fov
    fov_v = 26.8
    fov_h = 60
    resolution = 64
    vert_res = np.radians(fov_v/resolution) * 1000

    lidar_hor_mrad = 1.570796 #0.09 deg - velodyne hdl-64e lidar
    lidar_ver_mrad = vert_res #26.8/64 = ~0.42 deg - velodyne hdl-64e lidar

    #----------------------------------

    image_width = depthImage.shape[1]
    image_height = depthImage.shape[0]


    lidar_hor_deg = math.degrees(lidar_hor_mrad/1000)
    lidar_ver_deg = math.degrees(lidar_ver_mrad/1000)


    x_coords = np.arange(image_width)
    x_coords = np.repeat(x_coords[np.newaxis,:], image_height, 0)

    y_coords = np.arange(image_height)
    y_coords = np.repeat(y_coords[np.newaxis,:], image_width, 0).transpose()
    
    x_coords_lidar = np.arange(-(fov_h/2),(fov_h/2), lidar_hor_deg)
    y_coords_lidar = np.arange(-(fov_v/2),(fov_v/2), lidar_ver_deg)
    
    x_size = x_coords_lidar.size
    y_size = y_coords_lidar.size
    
    x_coords_lidar = np.repeat(x_coords_lidar[np.newaxis,:], y_size, 0)
    y_coords_lidar = np.repeat(y_coords_lidar[np.newaxis,:], x_size, 0).transpose()


    bool_mat = np.full((image_height,image_width), False)

    r = 1

    #x=x
    #y=z
    #z=y

    x = r * np.sin(np.deg2rad(90 + y_coords_lidar)) * np.cos(np.deg2rad(90 + x_coords_lidar))
    z = r * np.sin(np.deg2rad(90 + y_coords_lidar)) * np.sin(np.deg2rad(90 + x_coords_lidar))
    y = r * np.cos(np.deg2rad(90 + y_coords_lidar))

    n = np.array((0,0,1))
    p0 = np.array((0,0,f))
    l0 = np.array((0,0,0))

    l = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1), y.reshape(y.shape[0] * y.shape[1], 1), z.reshape(z.shape[0] * z.shape[1], 1)))
    
    p0l0 = p0 - l0
    dot_p0_n = np.dot(p0l0, n)
    ln = np.dot(l, n)

    d = dot_p0_n/ln
    d = d[:,np.newaxis]

    p = l * d

    y_ = p[:,1] + py
    x_ = p[:,0] + px

    y_ = y_.round().astype(int)
    x_ = x_.round().astype(int)

    #get indices of points which lie on the image -> points lying outside of the image get discarded
    indices = np.where((x_>=0)*(x_<image_width))
    x_ = np.take(x_, indices)
    y_ = np.take(y_, indices)

    indices = np.where((y_>=0)*(y_<image_height))
    x_ = np.take(x_, indices[1])
    y_ = np.take(y_, indices[1])

    bool_mat[y_, x_] = True
    
    
    red = np.array(redChannel)
    green = np.array(greenChannel)

    Z_mat = (norm(red) + norm(green)/255.0) * 655.36

    X_mat = ((x_coords - px) * Z_mat)/f
    Y_mat = ((y_coords - py) * Z_mat)/f

    X_mat = X_mat[bool_mat]
    Y_mat = Y_mat[bool_mat]
    Z_mat = Z_mat[bool_mat]

    z_indices = np.where(Z_mat <= max_distance)
    
    X_mat = X_mat[z_indices]
    Y_mat = Y_mat[z_indices]
    Z_mat = Z_mat[z_indices]

    pointsarr = np.hstack((X_mat.reshape(X_mat.shape[0], 1), Y_mat.reshape(Y_mat.shape[0], 1), Z_mat.reshape(Z_mat.shape[0], 1)))
   
    
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), engine=None, size=(600, 600))
    G = mlab.points3d(pointsarr[:, 0], pointsarr[:, 1], pointsarr[:, 2], colormap='gnuplot', scale_factor=0.03, figure=fig)
    
    
    object_ground_truth_lines = []
    with open(object_ground_truthPath) as object_ground_truthFile:
        object_ground_truth_lines = object_ground_truthFile.readlines()

    object_ground_truths = {}
    kitti_data = []



    for line in object_ground_truth_lines:
        object_ground_truths[line.split()[1]] = [line.split()[2], [line.split()[11],line.split()[10],line.split()[12]], line.split()[13:16], line.split()[16], line.split()[20:23]]

        kitti_data.append([line.split()[2], line.split()[3], line.split()[4], line.split()[5], line.split()[6:10],line.split()[10:13], line.split()[13:16], line.split()[16]])


    objects = ['Coupe', 'Truck', 'SUV', 'Bus', 'Hatchback', 'Cyclist', 'Sedan', 'Van', 'Motorcyclist', 'PickupTruck', 'Pedestrian']



    for obj in object_ground_truths.values(): 

        obj_category = obj[0]

        if not (obj_category in objects):

            continue

        obj_dimensions = np.array(obj[1], dtype=float)
        obj_center = np.array(obj[2], dtype=float)

        if obj_center[2] > 150:
            continue

        obj_y_rotation = obj[3]
        obj_rotation = np.array(obj[4], dtype=float)
        
        #create corner points of bounding box (back)
        left_back_upper = np.array(obj_center - (obj_dimensions / 2) ,dtype=float)
        left_back_lower = np.array(obj_center + [-obj_dimensions[0]/2, obj_dimensions[1]/2, -obj_dimensions[2]/2] ,dtype=float)
        right_back_upper = np.array(obj_center + [obj_dimensions[0]/2, -obj_dimensions[1]/2, -obj_dimensions[2]/2] ,dtype=float)
        right_back_lower = np.array(obj_center + [obj_dimensions[0]/2, obj_dimensions[1]/2, -obj_dimensions[2]/2] ,dtype=float)
        #create corner points of bounding box (front)
        left_front_upper = np.array(obj_center + [-obj_dimensions[0]/2, -obj_dimensions[1]/2, obj_dimensions[2]/2] ,dtype=float)
        left_front_lower = np.array(obj_center + [-obj_dimensions[0]/2, obj_dimensions[1]/2, obj_dimensions[2]/2] ,dtype=float)
        right_front_upper = np.array(obj_center + [obj_dimensions[0]/2, -obj_dimensions[1]/2, obj_dimensions[2]/2] ,dtype=float)
        right_front_lower = np.array(obj_center + [obj_dimensions[0]/2, obj_dimensions[1]/2, obj_dimensions[2]/2] ,dtype=float)
        
        color=(0, 1, 0)
        tube_radius=None
        line_width=2
        
        #draw lines of bounding box
        
        #back square
        mlab.plot3d([left_back_lower[0],right_back_lower[0]],[left_back_lower[1],right_back_lower[1]],[left_back_lower[2],right_back_lower[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([left_back_lower[0],left_back_upper[0]],[left_back_lower[1],left_back_upper[1]],[left_back_lower[2],left_back_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([left_back_upper[0],right_back_upper[0]],[left_back_upper[1],right_back_upper[1]],[left_back_upper[2],right_back_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([right_back_lower[0],right_back_upper[0]],[right_back_lower[1],right_back_upper[1]],[right_back_lower[2],right_back_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        #front square
        mlab.plot3d([left_front_lower[0],right_front_lower[0]],[left_front_lower[1],right_front_lower[1]],[left_front_lower[2],right_front_lower[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([left_front_lower[0],left_front_upper[0]],[left_front_lower[1],left_front_upper[1]],[left_front_lower[2],left_front_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([left_front_upper[0],right_front_upper[0]],[left_front_upper[1],right_front_upper[1]],[left_front_upper[2],right_front_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([right_front_lower[0],right_front_upper[0]],[right_front_lower[1],right_front_upper[1]],[right_front_lower[2],right_front_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        
        #connection lines
        mlab.plot3d([left_front_lower[0],left_back_lower[0]],[left_front_lower[1],left_back_lower[1]],[left_front_lower[2],left_back_lower[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([left_back_upper[0],left_front_upper[0]],[left_back_upper[1],left_front_upper[1]],[left_back_upper[2],left_front_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([right_back_upper[0],right_front_upper[0]],[right_back_upper[1],right_front_upper[1]],[right_back_upper[2],right_front_upper[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
        
        mlab.plot3d([right_front_lower[0],right_back_lower[0]],[right_front_lower[1],right_back_lower[1]],[right_front_lower[2],right_back_lower[2]],color=color,    tube_radius=tube_radius,line_width=line_width, figure=fig)
      
      
    #visualize everything in viewer
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    mlab.show(stop=True)

main()



