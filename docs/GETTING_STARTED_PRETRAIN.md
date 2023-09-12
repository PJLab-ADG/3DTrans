
# Getting Started for Point Cloud Pre-training

Please download and preprocess the point cloud datasets according to the [dataset guidance](GETTING_STARTED_DB.md)

## :fire: News of Our 3D Pre-training Study
- The code of AD-PT will be released (updated on Sep. 2023).
- 3DTrans has supported the Autonomous Driving Pre-training using the [PointContrast](https://arxiv.org/abs/2007.10985) 
<!-- - We are exploring the scalable pre-training solution by continuously increasing the scales of 3D pre-training data, and if you are interested in this topic, do not hesitate to contact me (bo.zhangzx@gmail.com). -->

### Pre-training ONCE using PointContrast 

* a) Train PV-RCNN++ backbone with [PointContrast](https://arxiv.org/abs/2007.10985) using multiple GPUs
```shell script
sh scripts/PRETRAIN/dist_train_pointcontrast.sh ${NUM_GPUs} \
--cfg_file ./cfgs/once_models/unsupervised_model/pointcontrast_pvrcnn_res_plus_backbone.yaml \
--batch_size 4 \
--epochs 30
```

or 

* a) Train PV-RCNN++ backbone with [PointContrast](https://arxiv.org/abs/2007.10985) using multiple machines
```shell script
sh scripts/PRETRAIN/slurm_train_pointcontrast.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/once_models/unsupervised_model/pointcontrast_pvrcnn_res_plus_backbone.yaml \
--batch_size 4 \
--epochs 30
```

* b) Fine-tuning PV-RCNN++ on other 3D datasets such as Waymo
> Note that you need to set the `--pretrained_model ${PRETRAINED_MODEL}` using the checkpoint obtained in the Pre-training phase.
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ${CONFIG_FILE} \
--batch_size ${BATCH_SIZE} \
--pretrained_model ${PRETRAINED_MODEL} 
```

## AD-PT Results:

We report the fine-tuning results using our AD-PT backbones.


### Fine-tuning Results on Waymo:

|                                                                                      | Data amount | Overall | Vehicle                | Pedestrian | Cyclist |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- | -----|
| [SECOND (From scratch)]()              | 3%  |   52.00 / 37.70 | 58.11 / 57.44 | 51.34 / 27.38 | 46.57 / 28.28  |
| [SECOND (AD-PT)]()                     | 3%  |   55.41 / 51.78 | 60.53 / 59.93 | 54.91 / 45.78 | 50.79 / 49.65  |
| [SECOND (From scratch)]()              | 20% |   60.62 / 56.86 | 64.26 / 63.73 | 59.72 / 50.38 | 57.87 / 56.48  |
| [SECOND (AD-PT)]()                     | 20% |   61.26 / 57.69 | 64.54 / 64.00 | 60.25 / 51.21 | 59.00 / 57.86  |
| [CenterPoint (From scratch)]()         | 3%  |   59.00 / 56.29 | 57.12 / 56.57 | 58.66 / 52.44 | 61.24 / 59.89  |
| [CenterPoint (AD-PT)]()                | 3%  |   61.21 / 58.46 | 60.35 / 59.79 | 60.57 / 54.02 | 62.73 / 61.57  |
| [CenterPoint (From scratch)]()         | 20% |   66.47 / 64.01 | 64.91 / 64.42 | 66.03 / 60.34 | 68.49 / 67.28  |
| [CenterPoint (AD-PT)]()                | 20% |   67.17 / 64.65 | 65.33 / 64.83 | 67.16 / 61.20 | 69.39 / 68.25  |
| [PV-RCNN++ (From scratch)]()           | 3%  |   63.81 / 61.10 | 64.42 / 63.93 | 64.33 / 57.79 | 62.69 / 61.59  |
| [PV-RCNN++ (AD-PT)]()                  | 3%  |   68.33 / 65.69 | 68.17 / 67.70 | 68.82 / 62.39 | 68.00 / 67.00  |
| [PV-RCNN++ (From scratch)]()           | 20% |   69.97 / 67.58 | 69.18 / 68.75 | 70.88 / 65.21 | 69.84 / 68.77  |
| [PV-RCNN++ (AD-PT)]()                  | 20% |   71.55 / 69.23 | 70.62 / 70.19 | 72.36 / 66.82 | 71.69 / 70.70  |

### Fine-tuning Results on nuScenes:

|                                                                                      | Data amount | mAP | NDS | Car | Truck | CV. | Bus | Trailer | Barrier | Motorcycle | Bicycle | Pedestrian | Cyclist |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- | -----| --- | --- | --- | --- | --- | --- | --- | --- |
| [SECOND (From scratch)]() | 5% | 29.24 | 39.74 | 67.69 | 33.02 | 7.15 | 45.91 | 17.67 | 25.23 | 11.92 | 0.00 | 53.00 | 30.74 |
| [SECOND (AD-PT)]()    | 5% | 37.69 | 47.95 | 74.89 | 41.82 | 12.05 | 54.77 | 28.92 | 34.41 | 23.63 | 3.19 | 63.61 | 39.54 |
| [SECOND (From scratch)]() | 100% | 50.59 | 62.29 | - | - | - | - | - | - | - | - | - | - |
| [SECOND (AD-PT)]()        | 100% | 52.23 | 63.04 | 83.12 | 52.86 | 15.24 | 68.58 | 37.54 | 59.48 | 46.01 | 20.44 | 78.96 | 60.05 |
| [CenterPoint (From scratch)]()  | 5%   | 42.68 | 50.41 | 77.82 | 43.61 | 10.65 | 44.01 | 18.71 | 52.95 | 36.26 | 16.76 | 37.62 | 54.52 |
| [CenterPoint (AD-PT)]()         | 5%   | 44.99 | 52.99 | 78.90 | 43.82 | 11.13 | 55.16 | 21.22 | 55.10 | 39.03 | 17.76 | 72.28 | 55.43 |
| [CenterPoint (From scratch)]()  | 100% | 56.2  | 64.5  | 84.8  | 53.9  | 16.8  | 67.0  | 35.9  | 64.8  | 55.8  | 36.4  | 83.1  | 63.4  |
| [CenterPoint (AD-PT)]()         | 100% | 57.17 | 65.48 | 84.86 | 54.37 | 16.09 | 67.354 | 36.06 | 64.31 | 58.50 |40.58 | 83.53 | 66.05 |

### Fine-tuning Results on KITTI:

|                                                                                      | Data amount | mAP ( Mod.) | Car (mod.)               | Pedestrian (Mod.) | Cyclist (Mod.) |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- | -----|
| [SECOND (From scratch)]() | 20%  | 61.70 | 78.83 | 47.23 | 59.06 |
| [SECOND (AD-PT)]()        | 20%  | 65.95 | 80.70 | 49.67 | 67.50 |
| [SECOND (From scratch)]() | 100% | 66.70 | 80.78 | 52.61 | 66.71 |
| [SECOND (AD-PT)]()        | 100% | 67.58 | 81.39 | 53.58 | 67.78 |
| [PV-RCNN (From scratch)]()| 20%  | 66.71 | 82.52 | 53.33 | 64.28 |
| [PV-RCNN (AD-PT)]()       | 20%  | 69.43 | 82.75 | 57.59 | 67.96 |
| [PV-RCNN (From scratch)]()| 100% | 70.57 | 84.50 | 57.06 | 70.14 |
| [PV-RCNN (AD-PT)]()       | 100% | 73.01 | 84.75 | 60.79 | 73.49 |


&ensp;
## Pre-training using AD-PT

We are actively exploring the possibility of boosting the **3D pre-training generalization ability**. The corresponding code is **coming soon** in 3DTrans-v0.2.0.

- **AD-PT pre-trained checkpoints**
  <span id="once-ckpt">
  
  |  Pre-training Method | Pre-trained data | Pre-trained model |
  | ---------------- | ---------------- | ----------------- |
  | AD-PT | ONCE PS-100K     | [once-100K-ckpt](https://drive.google.com/file/d/1MG7rZu19oFHi2fZs4xA_Ts1tMzPV8yEi/view?usp=sharing)|
  | AD-PT | ONCE PS-500K     | [once-500K-ckpt](https://drive.google.com/file/d/1PV2K0J6geK5BkDbG6-XiPvWOW60lN41S/view?usp=sharing) |
  | AD-PT  | ONCE PS-1M       | [once-1M-ckpt](https://drive.google.com/file/d/13WD7sjXkZ0tYxIgM8DrMKvBOT9Q85YPf/view?usp=sharing) |