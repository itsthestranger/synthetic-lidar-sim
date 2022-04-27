generate_dataset.py:
	- python file to generate synthetic dataset from Apollo Synthetic dataset (https://apollo.auto/synthetic.html)
	- Depth and Object_GroundTruth folders from Apollo Synthetic dataset needed, need to be in the same main folder
	- constants like FoV or LiDAR-resolution are defined in the 'gen_pc'-function
	- to run, place generate_dataset.py in the same folder, where the 'Depth' and 'Object_GroundTruth' folder of the 
	Apollo Synthetic dataset reside and execute it (requires Python >= 3.5)
	- running generate_dataset.py creates a dataset folder with the three subfolders 'depthImg', 'label' and 'pc'
	'pc' holds the point cloud files, 'label' the corresponding ground truth labels and 'depthImg' holds the corresponding depth images (depth images are not needed
	for the training process, but can be useful when reviewing results)
	

depthImg_to_lidar_pc_with_boundingBoxes_o3d.py:
	- python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the open3d library


depthImg_to_lidar_pc_with_boundingBoxes_mlab.py:
	- python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the mayavi library
	
	

*This project utilizes the OpenPCDet project (https://github.com/open-mmlab/OpenPCDet).