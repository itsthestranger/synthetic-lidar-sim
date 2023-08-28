<h1>LiDAR resolution simulation in synthetic
training data for 3D object detection</h1>

<h2>Abstract</h2>
3D object detectors need large amounts of annotated training samples to reach high
accuracies, but generating these annotated training samples is expensive and labouri-
ouse. This is the incentive to try to generate these datasets synthetically. Synthetic
data describes data taken from virual environments like game engines. In this work we
generate point clouds from the depth images of the Apollo Synthetic dataset to build
a dataset on which a 3D object detector is trained. We show the impact the correct
simulation of the LiDAR has on the accuracy of the 3D object detector. Additionally
we simulate real world effects like noise and drop-out to further increase the accuracy
of the 3D object detector.
<br />

<h2>Overview</h2>
<p>Source files of the practical part of my bachelor thesis focusing on LiDAR resolution simulation in synthetic
training data for 3D object detection. More specifically a LiDAR is simulated on the Apollo Synthetic dataset (https://apollo.auto/synthetic.html) to create synthetic traning data, which can be utilized for 3D object detection.<br />
Further a model is trained on the synthetic dataset by utilizing the OpenPCDet codebase (https://github.com/open-mmlab/OpenPCDet).
</p>
<br />

<h2>Files</h2>

<h3>generate_dataset.py:</h3>
<ul>
  <li>python file to generate synthetic dataset from Apollo Synthetic dataset</li>
  <li>Depth and Object_GroundTruth folders from Apollo Synthetic dataset needed, need to be in the same main folder</li>
  <li>constants like FoV or LiDAR-resolution are defined in the &#39;gen_pc&#39;-function</li>
  <li>to run, place generate_dataset.py in the same folder, where the &#39;Depth&#39; and &#39;Object_GroundTruth&#39; folder of the Apollo Synthetic dataset reside and execute it (requires Python &gt;= 3.5)</li>
  <li>running generate_dataset.py creates a dataset folder with the three subfolders &#39;depthImg&#39;, &#39;label&#39; and &#39;pc&#39; &#39;pc&#39; holds the point cloud files, &#39;label&#39; the corresponding ground truth labels and &#39;depthImg&#39; holds the corresponding depth images (depth images are not needed for the training process, but can be useful when reviewing results)</li>
</ul>


<h3>get_anchor_sizes.py:</h3>
<ul>
  <li>python file to generate the average bounding box sizes (anchor sizes) for the classes 'Car', 'Pedestrian' and 'Cyclist'</li>
  <li>run and anchor sizes are printed out to the console</li>
</ul>


<h3>depthImg_to_lidar_pc_with_boundingBoxes_o3d.py:</h3>
<ul>
  <li>python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the open3d library</li>
</ul>


<h3>depthImg_to_lidar_pc_with_boundingBoxes_mlab.py:</h3>
<ul>
  <li>python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the mayavi library</li>
</ul>
<br />

<h2>Code Changes</h2>
<h3>Code changes to allow custom dataset:</h3>
<ol>
  <li>pcdet/datasets/kitti -&gt; duplicate kitti_dataset.py -&gt; rename to e.g. custom_dataset.py -&gt; in custom_dataset.py rename class to CustomDataset &amp; adjust data_path/save_path to custom dataset at the end of custom_dataset.py (example for data_path: data_path=ROOT_DIR / &#39;data&#39; / &#39;custom&#39;, with &#39;custom&#39; being the folder holding the custom dataset)</li>
  <li>
    pcdet/datasets -&gt; add custom dataset in __init__.py (marked in code snipped below)
  </li>

```python
    #code snipped of __init__.py including changes
	  from .kitti.kitti_dataset import KittiDataset

		#--- newly added custom dataset ---
		from .kitti.custom_dataset import CustomDataset
		#----------------------------------

		from .nuscenes.nuscenes_dataset import NuScenesDataset
		from .waymo.waymo_dataset import WaymoDataset

		__all__ = {
		    'DatasetTemplate': DatasetTemplate,
		    'KittiDataset': KittiDataset,
		    
		    #--- newly added custom dataset ---
		    'CustomDataset': CustomDataset,
		    #----------------------------------
		    
		    'NuScenesDataset': NuScenesDataset,
		    'WaymoDataset': WaymoDataset
		}
	
	  #--------------------------------------------------------------
```

  <li>
    tools/cfgs/dataset_configs
    <ul>
      <li>duplicate kitti_dataset.yaml</li>
      <li>rename to e.g. custom_dataset.yaml</li>
      <li>in custom_dataset.yaml change DATA_PATH to &#39;../data/custom&#39; (custom being the folder holding the custom dataset)</li>
    </ul>
  </li>
  
  <li>
    tools/cfgs/kitti_models
    <ul>
      <li>duplicate model e.g. pointpillar.yaml</li>
      <li>rename to e.g. pointpillar_custom.yaml</li>
      <li>in pointpillar_custom.yaml adjust _BASE_CONFIG_ to cfgs/dataset_configs/custom_dataset.yaml</li>
      <li>change USE_ROAD_PLANE to False if no road plane data is available for the custom dataset</li>
      <li>change the anchor sizes for each class at DENSE_HEAD -&gt; ANCHOR_GENERATOR_CONFIG to the anchor sizes corresponding to the custom dataset (average bounding box size per class)</li>
    </ul>
  </li>

</ol>
<br />

<h3>Changes to include real world effects:</h3>
<ul>
  <li>
    pcdet/datasets/kitti/custom_dataset.py -&gt; add simulation for real world effects in getitem function -&gt; adjusted __getitem__ function:
  </li>
  
  ```python
    #code snipped of custom_dataset.py including changes
	
		def __getitem__(self, index):
			# index = 4
			if self._merge_all_iters_to_one_epoch:
			    index = index % len(self.kitti_infos)

			info = copy.deepcopy(self.kitti_infos[index])

			sample_idx = info['point_cloud']['lidar_idx']

			points = self.get_lidar(sample_idx)

			#--- real world effects start (only introduced during training) ---
			if self.training:
			    #--- noise ---
			    #add 1cm noise
			    points[:,0:3] += np.random.normal(0.0, 0.01, size=points[:,0:3].shape)
			    #-------------
			    
			    #-- drop-out --
			    #remove random 30% of points
			    indices = np.arange(len(points))
			    indices = np.random.permutation(indices)
			    indices = indices[:int(len(indices) * 0.70)]
			    points = points[indices]
			    #--------------
			
			#--------------- real world effects simulation end ---------------
			
			
			calib = self.get_calib(sample_idx)

			.
			.
			.
			
			return data_dict
			
      #--------------------------------------------------------------

  ```
</ul>
<br />

<h3>Changes regarding to the visualization of point clouds:</h3>
<ol>
  <li>
    Only show bounding boxes with prediction score above 50 %:
  </li>
    <ul>
      <li>
        In OpenPCDet/tools/demo.py main function get only relevant points before visualizing:
      </li>
    </ul>

  ```python 
      #code snipped of demo.py with change marked
			
			.
			.
			.
			
			load_data_to_gpu(data_dict)
			pred_dicts, _ = model.forward(data_dict)

			#addition to only get relevant points -> score > 50%
			relevant_points = np.where(pred_dicts[0]['pred_scores'].cpu().numpy() > 0.5)
			#----------------------------------------------------
			
			V.draw_scenes(points=data_dict['points'][:, 1:], 
				ref_boxes=pred_dicts[0]['pred_boxes'][relevant_points],
				ref_scores=pred_dicts[0]['pred_scores'][relevant_points], 
				ref_labels=pred_dicts[0]['pred_labels'][relevant_points]
			)
			
			.
			.
			.
		
		#--------------------------------------------------
```

  <li>
    Increase the size of the point cloud points in the visualizer for nicer object representation:
  </li>
    <ul>
      <li>
        In OpenPCDet/tools/visual_utils/visualize_utils.py visualize_pts function change scale factor:
      </li>
    </ul>
    
  ```python 
      #code snipped of demo.py with change marked
			
			.
			.
			.
			
			load_data_to_gpu(data_dict)
			pred_dicts, _ = model.forward(data_dict)

			#addition to only get relevant points -> score > 50%
			relevant_points = np.where(pred_dicts[0]['pred_scores'].cpu().numpy() > 0.5)
			#----------------------------------------------------
			
			V.draw_scenes(points=data_dict['points'][:, 1:], 
				ref_boxes=pred_dicts[0]['pred_boxes'][relevant_points],
				ref_scores=pred_dicts[0]['pred_scores'][relevant_points], 
				ref_labels=pred_dicts[0]['pred_labels'][relevant_points]
			)
			
			.
			.
			.
		
		  #--------------------------------------------------
```

</ol>
<br />


<h2>Dataset Preparation</h2>
<h3>Custom dataset preparation to use with OpenPCDet:</h3>
<h4>Dataset needs to be adjusted to fit KITTI format:</h4>
<ul>
  <li>
    point cloud data: point cloud data need to be adjusted according to point 3. of the Demo.md in the OpenPCDet docs (https://github.com/open-mmlab/OpenPCDet/blob/master/docs/DEMO.md) i. e. point cloud axis might need to be adjusted to fit KITTI format
  </li>
  <li>
    dataset structure: dataset structure of custom dataset needs to follow the format of the kitti dataset which can be found under OpenPCDet/data/kitti in the OpenPCDet framework -&gt; the kitti folder consits of three subfolders (&#39;ImageSets&#39;, &#39;testing&#39;, &#39;training&#39;)
  </li>
  <ol>
    <li>
      OpenPCDet/data/kitti/ImageSets:
      <ul>
        <li>
          test.txt -&gt; list of all frames in the dataset
        </li>
        <li>
          train.txt -&gt; list of all frames on which the model is trained
        </li>
        <li>
          val.txt -&gt; list of all frames on which the model is evaluated during the training process
        </li>
      </ul>
    </li>
    <li>
      OpenPCDet/data/kitti/testing:
      <ul>
        <li>
          calib -&gt; folder holding the camera calibration for each frame
        </li>
        <li>
          image_2 -&gt; folder hodling the 2D image for each frame
        </li>
        <li>
          velodyne -&gt; folder holding the point cloud for each frame - all files in the calib/image_2/velodyne folders are named according to the frames defined in ImageSets/val.txt
        </li>
      </ul>
    </li>
    <li>
      OpenPCDet/data/kitti/training:
      <ul>
        <li>
          calib -&gt; folder holding the camera calibration for each frame
        </li>
        <li>
          image_2 -&gt; folder hodling the 2D image for each frame
        </li>
        <li>
          label_2 -&gt; folder holding the labels/object ground truths for each frame
        </li>
        <li>
          velodyne -&gt; folder holding the point cloud for each frame - all files in the calib/image_2/velodyne folders are named according to the frames defined in ImageSets/val.txt
        </li>
      </ul>
    </li>
  </ol>
</ul>

<br />
<h4>The custom dataset needs to be prepared to fit the above structure/data preperation of the kitti dataset</h4>

<h4>TODO:</h4>
<ul>
  <li>
    rename custom dataset frames to fit the kitti format (6-digit counter with leading zeros)
  </li>
  <li>
    copy most of custom dataset to training folder (point clouds go into velodyne folder and labels in label_2 folder) as only the data in the training folder is used for the training process -&gt; for 3D object detection calib and image_2 can be filled with pseudo-data, this means they only need to hold pseudo files with the correct names (for each point cloud a label file, image file and calib file with the same name needs to be available)
  </li>
  <li>
    adjust names in ImageSets to the according frames present in the dataset
  </li>
</ul>
<br />

<h2>Training</h2>
<h3>Train custom dataset:</h3>
<ol>
  <li>
    create data infos -&gt; GoTo OpenPCDet main directory and call: python -m pcdet.datasets.kitti.custom_dataset create_kitti_infos tools/cfgs/dataset_configs/custom_dataset.yaml
  </li>
  <li>
    start the training process -&gt; GoTo OpenPCDet/tools and call: python train.py --cfg_file cfgs/kitti_models/pointpillar_custom.yaml
  </li>
</ol>


