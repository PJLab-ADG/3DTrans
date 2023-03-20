# Getting Started
All dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs). 

UDA model configs are located within [tools/cfgs/DA](../tools/cfgs/DA) for different datasets. 

ADA model configs are located within [tools/cfgs/ADA](../tools/cfgs/ADA) for different datasets. 

MDF model configs are located within [tools/cfgs/MDF](../tools/cfgs/MDF) for different datasets. 

SSDA model configs are located within [tools/cfgs/SSDA](../tools/cfgs/SSDA) for different datasets. 


Other model configs are located within [tools/cfgs/waymo_models](../tools/cfgs/waymo_models), [tools/cfgs/nuscenes_models](../tools/cfgs/nuscenes_models), [tools/cfgs/kitti_models](../tools/cfgs/kitti_models) for different datasets. 


&ensp;
# How to train our model using Ceph
* Install petrel_client to enable the Ceph / PetrelBackend

* Add '~/.petreloss.conf', which is a config file of Ceph, saving your KEY/ACCESS_KEY of S3 Ceph

* You need to ensure that the file organization of all datasets in the Ceph (Petrel-OSS) is consistent with that described above.

* For dataset config files, such as tools/cfgs/dataset_configs/waymo/OD/waymo_dataset.yaml, uncomment OSS_PATH and add the s3://path_of_your_dataset as follows:
```shell script
OSS_PATH: 's3://${PATH_TO_DATASET}/waymo_0.5.0'
```

&ensp;
## Dataset Preparation

We provide the dataloader for lyft, ONCE, waymo, KITTI and NuScenes datasets.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti/OD/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
3DTrans
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti/OD/kitti_dataset.yaml
```

&ensp;
### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
3DTrans
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes/OD/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

&ensp;
### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
3DTrans
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-0-0
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo/OD/waymo_dataset.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 

&ensp;
### ONCE Dataset
* Please download the official ONCE dataset [website](https://once-for-auto-driving.github.io/download.html#downloads). Please follow the instructions to unzip and organize the data.

* Please organize the data as follows:
```
ONCE_Benchmark
├── data
│   ├── once
│   │   │── ImageSets
|   |   |   ├──train.txt
|   |   |   ├──val.txt
|   |   |   ├──test.txt
|   |   |   ├──raw_small.txt (100k unlabeled)
|   |   |   ├──raw_medium.txt (500k unlabeled)
|   |   |   ├──raw_large.txt (1M unlabeled)
│   │   │── data
│   │   │   ├──000000
|   |   |   |   |──000000.json (infos)
|   |   |   |   |──lidar_roof (point clouds)
|   |   |   |   |   |──frame_timestamp_1.bin
|   |   |   |   |  ...
|   |   |   |   |──cam0[1-9] (images)
|   |   |   |   |   |──frame_timestamp_1.jpg
|   |   |   |   |  ...
|   |   |   |  ...
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once/OD/once_dataset.yaml
```


&ensp;
### Lyft Dataset
* Please download the official [Lyft Level5 perception dataset](https://level-5.global/data/perception) and 
organize the downloaded files as follows: 
```
3DTrans
├── data
│   ├── lyft
│   │   │── ImageSets
│   │   │── trainval
│   │   │   │── data & maps(train_maps) & images(train_images) & lidar(train_lidar) & train_lidar
│   │   │── test
│   │   │   │── data & maps(test_maps) & test_images & test_lidar
├── pcdet
├── tools
```

* Install the `lyft-dataset-sdk` with version `0.0.8` by running the following command: 
```shell script
pip install -U lyft_dataset_sdk==0.0.8
```

* Generate the training & validation data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.lyft.lyft_dataset --func create_lyft_infos \
    --cfg_file tools/cfgs/dataset_configs/lyft/OD/lyft_dataset.yaml
```
* Generate the test data infos by running the following command: 
```python 
python -m pcdet.datasets.lyft.lyft_dataset --func create_lyft_infos \
    --cfg_file tools/cfgs/dataset_configs/lyft/OD/lyft_dataset.yaml --version test
```

* You need to check carefully since we don't provide a benchmark for it.

