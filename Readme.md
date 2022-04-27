<p>generate_dataset.py:<br />
&nbsp;&nbsp; &nbsp;- python file to generate synthetic dataset from Apollo Synthetic dataset<br />
&nbsp;&nbsp; &nbsp;- Depth and Object_GroundTruth folders from Apollo Synthetic dataset needed, need to be in the same main folder<br />
&nbsp;&nbsp; &nbsp;- constants like FoV or LiDAR-resolution are defined in the &#39;gen_pc&#39;-function<br />
&nbsp;&nbsp; &nbsp;- to run, place generate_dataset.py in the same folder, where the &#39;Depth&#39; and &#39;Object_GroundTruth&#39; folder of the<br />
&nbsp;&nbsp; &nbsp;Apollo Synthetic dataset reside and execute it (requires Python &gt;= 3.5)<br />
&nbsp;&nbsp; &nbsp;- running generate_dataset.py creates a dataset folder with the three subfolders &#39;depthImg&#39;, &#39;label&#39; and &#39;pc&#39;<br />
&nbsp;&nbsp; &nbsp;&#39;pc&#39; holds the point cloud files, &#39;label&#39; the corresponding ground truth labels and &#39;depthImg&#39; holds the corresponding depth images (depth images are not needed<br />
&nbsp;&nbsp; &nbsp;for the training process, but can be useful when reviewing results)<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;<br />
get_anchor_sizes.py:<br />
&nbsp;&nbsp; &nbsp;- python file to generate the average bounding box sizes (anchor sizes) for the classes &#39;Car&#39;, &#39;Pedestrian&#39; and &#39;Cyclist&#39;<br />
&nbsp;&nbsp; &nbsp;- run and anchor sizes are printed out to the console</p>

<p>&nbsp;</p>

<p>depthImg_to_lidar_pc_with_boundingBoxes_o3d.py:<br />
&nbsp;&nbsp; &nbsp;- python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the open3d library</p>

<p><br />
depthImg_to_lidar_pc_with_boundingBoxes_mlab.py:<br />
&nbsp;&nbsp; &nbsp;- python file to generate and visualize a point cloud with bounding boxes for the objects from a specific depth image with the mayavi library</p>

<p>&nbsp;</p>

<p>------------- CODE CHANGES -------------</p>

<p>Code changes to allow custom dataset:</p>

<p>&nbsp;&nbsp; &nbsp;1.) pcdet/datasets/kitti -&gt; duplicate kitti_dataset.py -&gt; rename to e.g. custom_dataset.py -&gt; in custom_dataset.py rename class to<br />
&nbsp;&nbsp; &nbsp;CustomDataset &amp; adjust data_path/save_path to custom dataset at the end of custom_dataset.py (example for data_path: data_path=ROOT_DIR / &#39;data&#39; / &#39;custom&#39;&nbsp;&nbsp; ,<br />
&nbsp;&nbsp; &nbsp;with &#39;custom&#39; being the folder holding the custom dataset)</p>

<p>&nbsp;&nbsp; &nbsp;2.) pcdet/datasets -&gt; add custom dataset in __init__.py (marked in code snipped below)<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;------- code snipped of __init__.py including changes --------<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;from .kitti.kitti_dataset import KittiDataset</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;#--- newly added custom dataset ---<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;from .kitti.custom_dataset import CustomDataset<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;#----------------------------------</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;from .nuscenes.nuscenes_dataset import NuScenesDataset<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;from .waymo.waymo_dataset import WaymoDataset</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;__all__ = {<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &#39;DatasetTemplate&#39;: DatasetTemplate,<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &#39;KittiDataset&#39;: KittiDataset,<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #--- newly added custom dataset ---<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &#39;CustomDataset&#39;: CustomDataset,<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #----------------------------------<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &#39;NuScenesDataset&#39;: NuScenesDataset,<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &#39;WaymoDataset&#39;: WaymoDataset<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;}<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;--------------------------------------------------------------</p>

<p><br />
&nbsp;&nbsp; &nbsp;3.) tools/cfgs/dataset_configs<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- duplicate kitti_dataset.yaml<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- rename to e.g. custom_dataset.yaml<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- in custom_dataset.yaml change DATA_PATH to &#39;../data/custom&#39; (custom being the folder holding the custom dataset)</p>

<p><br />
&nbsp;&nbsp; &nbsp;4.) tools/cfgs/kitti_models<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- duplicate model e.g. pointpillar.yaml<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- rename to e.g. pointpillar_custom.yaml<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- in pointpillar_custom.yaml adjust _BASE_CONFIG_ to cfgs/dataset_configs/custom_dataset.yaml<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- change USE_ROAD_PLANE to False if no road plane data is available for the custom dataset<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- change the anchor sizes for each class at DENSE_HEAD -&gt; ANCHOR_GENERATOR_CONFIG to the anchor sizes corresponding to the custom dataset (average bounding box size per class)</p>

<p><br />
--- end of code changes for OpenPCDet to allow custom dataset ---</p>

<p><br />
changes to include real world effects:</p>

<p>&nbsp;&nbsp; &nbsp;1.) pcdet/datasets/kitti/custom_dataset.py -&gt; add simulation for real world effects in getitem function -&gt; adjusted __getitem__ function:<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;------- code snipped of custom_dataset.py including changes --------<br />
&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;def __getitem__(self, index):<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;# index = 4<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;if self._merge_all_iters_to_one_epoch:<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; index = index % len(self.kitti_infos)</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;info = copy.deepcopy(self.kitti_infos[index])</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;sample_idx = info[&#39;point_cloud&#39;][&#39;lidar_idx&#39;]</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;points = self.get_lidar(sample_idx)</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;#--- real world effects start (only introduced during training) ---<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;if self.training:<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #--- noise ---<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #add 1cm noise<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; points[:,0:3] += np.random.normal(0.0, 0.01, size=points[:,0:3].shape)<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #-------------<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #-- drop-out --<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #remove random 30% of points<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; indices = np.arange(len(points))<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; indices = np.random.permutation(indices)<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; indices = indices[:int(len(indices) * 0.70)]<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; points = points[indices]<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; #--------------<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;#--------------- real world effects simulation end ---------------<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;calib = self.get_calib(sample_idx)</p>

<p>&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;.<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;.<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;.<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;return data_dict<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --------------------------------------------------------------<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;<br />
--- end of changes to include real world effects ---</p>

<p>----------------------------------------</p>

<p><br />
Custom dataset preparation to use with OpenPCDet:</p>

<p>Dataset needs to be adjusted to fit KITTI format:<br />
&nbsp;&nbsp; &nbsp;- point cloud data: point cloud data need to be adjusted according to point 3. of the Demo.md in the OpenPCDet<br />
&nbsp;&nbsp; &nbsp;docs (https://github.com/open-mmlab/OpenPCDet/blob/master/docs/DEMO.md) i. e. point cloud axis might need to be adjusted to fit KITTI format</p>

<p>&nbsp;&nbsp; &nbsp;- dataset structure: dataset structure of custom dataset needs to follow the format of the kitti dataset which can be found under OpenPCDet/data/kitti<br />
&nbsp;&nbsp; &nbsp;in the OpenPCDet framework -&gt; the kitti folder consits of three subfolders (&#39;ImageSets&#39;, &#39;testing&#39;, &#39;training&#39;)</p>

<p>&nbsp;&nbsp; &nbsp;-&gt; OpenPCDet/data/kitti/ImageSets:<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- test.txt -&gt; list of all frames in the dataset<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- train.txt -&gt; list of all frames on which the model is trained<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- val.txt -&gt; list of all frames on which the model is evaluated during the training process</p>

<p>&nbsp;&nbsp; &nbsp;-&gt; OpenPCDet/data/kitti/testing:<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- calib -&gt; folder holding the camera calibration for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- image_2 -&gt; folder hodling the 2D image for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- velodyne -&gt; folder holding the point cloud for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;all files in the calib/image_2/velodyne folders are named according to the frames defined in ImageSets/val.txt</p>

<p>&nbsp;&nbsp; &nbsp;-&gt; OpenPCDet/data/kitti/training:<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- calib -&gt; folder holding the camera calibration for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- image_2 -&gt; folder hodling the 2D image for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- label_2 -&gt; folder holding the labels/object ground truths for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;- velodyne -&gt; folder holding the point cloud for each frame<br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;all files in the calib/image_2/label_2/velodyne folders are named according to the frames defined in ImageSets/training.txt</p>

<p><br />
The custom dataset needs to be prepared to fit the above structure/data preperation of the kitti dataset</p>

<p>TODO:<br />
&nbsp;&nbsp; &nbsp;- rename custom dataset frames to fit the kitti format (6-digit counter with leading zeros)<br />
&nbsp;&nbsp; &nbsp;- copy most of custom dataset to training folder (point clouds go into velodyne folder and labels in label_2 folder) as only the data<br />
&nbsp;&nbsp; &nbsp;in the training folder is used for the training process -&gt; for 3D object detection calib and image_2 can be filled with pseudo-data,<br />
&nbsp;&nbsp; &nbsp;this means they only need to hold pseudo files with the correct names (for each point cloud a label file, image file and calib file with the same name needs to be available)<br />
&nbsp;&nbsp; &nbsp;- adjust names in ImageSets to the according frames present in the dataset</p>

<p><br />
--- end of custom dataset preperation ---</p>

<p>&nbsp;</p>

<p>Train custom dataset:<br />
&nbsp;&nbsp; &nbsp;1.) create data infos -&gt; GoTo OpenPCDet main directory and call: python -m pcdet.datasets.kitti.custom_dataset create_kitti_infos tools/cfgs/dataset_configs/custom_dataset.yaml<br />
&nbsp;&nbsp; &nbsp;2.) start the training process -&gt; GoTo OpenPCDet/tools and call: python train.py --cfg_file cfgs/kitti_models/pointpillar_custom.yaml</p>

<p>*This project utilizes the OpenPCDet project (https://github.com/open-mmlab/OpenPCDet).</p>
