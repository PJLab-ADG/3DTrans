# Getting Started & Problem Definition

Different from UDA task, the purpose of an Active Domain Adaptation (ADA) task is to pick up a subset of unlabeled target domain $t$ to perform the manual annotation process, such that we can achieve a good trade-off between high performance and low annotation cost, where labeled training data from the source domain $s$ (such as point cloud or images) are assumed to be available for initializing the training model.

&ensp;
&ensp;
# Getting Started & Training-Testing for ADA setting

Here, We take Waymo-to-KITTI adaptation as an example.

## Pretraining stage: train the source-only model on the labeled source domain:

* Train FEAT=3 (X,Y,Z) with SN (statistical normalization) using multiple GPUs
  
  ```shell
  sh scripts/dist_train.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor_sn_kitti.yaml
  ```

* Train FEAT=3 (X,Y,Z) with SN (statistical normalization) using multiple machines
  
  ```shell
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor_sn_kitti.yaml
  ```

* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple GPUs
  
  ```shell
  sh scripts/dist_train.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml
  ```

* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple machines
  
  ```shell
  sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml
  ```

* Train other baseline detectors such as Voxel R-CNN using multiple GPUs
  
  ```shell
  sh scripts/dist_train.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml
  ```

* Train other baseline detectors such as Voxel R-CNN using multiple machines
  
  ```shell
  sh scripts/slurm_train.sh ${PARTITION} ${JOB} ${NUM_NODES} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml
  ```

## Evaluate the source-pretrained model:

* Note that for the cross-domain setting where the KITTI dataset is regarded as the target domain, please try --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True to enable front view point cloud only. We report the best model for all epochs on the validation set.

* Test the source-only models using multiple GPUs
  
  ```shell
  sh scripts/dist_test.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml \ 
   --ckpt ${CKPT} 
  ```

* Test the source-only models using multiple machines
  
  ```shell
  sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_NODES} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml \ 
   --ckpt ${CKPT}
  ```

* Test the source-only models of all ckpts using multiple GPUs
  
  ```shell
  sh scripts/dist_test.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml \ 
   --eval_all
  ```

* Test the source-only models of all ckpts using multiple machines
  
  ```shell
  sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_NODES} \ 
   --cfg_file ./cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml \ 
   --eval_all
  ```

## Bi3D Adaptation stage 1: active source domain data

* You need to set the `--pretrained_model ${PRETRAINED_MODEL}` when finish the pretraining model stage

* Train with SN (statistical normalization) using multiple GPUs
  
  ```shell
  sh scripts/ADA/dist_train_active_source.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_source_only.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```
- Train with SN (statistical normalization) using multiple machines
  
  ```shell
  sh scripts/ADA/slurm_train_active_source.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_source_only.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train without SN (statistical normalization) using multiple GPUs
  
  ```shell
  sh scripts/ADA/dist_train_active_source.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_source_only_wosn.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train without SN (statistical normalization) using multiple machines
  
  ```shell
  sh scripts/ADA/slurm_train_active_source.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_source_only_wosn.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

## Bi3D Adaptation stage 2: active target domain data

- You need to set the `--pretrained_model ${PRETRAINED_MODEL}` when finish the adaptation stage 1

- Train with 1% annotation budget using multiple GPUs
  
  ```shell
  sh scripts/ADA/dist_train_active.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_01.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train with 1% annotation budget using multiple machines
  
  ```shell
  sh scripts/ADA/slurm_train_active.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_01.yaml \  
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train with 5% annotation budget using multiple GPUs
  
  ```shell
  sh scripts/ADA/dist_train_active.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_05.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train with 5% annotation budget using multiple machines
  
  ```shell
  sh scripts/ADA/slurm_train_active.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_05.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

## Evaluating the model on the labeled target domain

* Test with a ckpt file: 
  
  ```shell
  python test.py --cfg_file ${CONFIG_FILE} \ 
   --batch_size ${BATCH_SIZE} \ 
   --ckpt ${CKPT}
  ```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
  
  ```shell
  python test.py \ 
   --cfg_file ${CONFIG_FILE} \ 
   --batch_size ${BATCH_SIZE} \ 
   --eval_all
  ```

* Notice that if you want to test on the setting with KITTI as **target domain**, 
  please add `--set DATA_CONFIG_TAR.FOV_POINTS_ONLY True` to enable front view point cloud only: 
  
  ```shell
  python test.py \ 
   --cfg_file ${CONFIG_FILE} \ 
   --batch_size ${BATCH_SIZE} \ 
   --eval_all \ 
   --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True
  ```

* To test with multiple machines for S-Proj:
  
  ```shell
  sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_NODES} \ 
    --cfg_file ${CONFIG_FILE} \ 
    --batch_size ${BATCH_SIZE}
  ```

## Train with other active domain adaptation / active learning methods

- Train with TQS
  
  ```shell
  sh scripts/ADA/dist_train_active_TQS.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_TQS.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train with CLUE
  
  ```shell
  sh scripts/ADA/dist_train_active_CLUE.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_CLUE.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

## Combine Bi3D and UDA

- Train with multiple GPUs
  
  ```shell
  sh scripts/ADA/dist_train_active_st3d.sh ${NUM_GPUs} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_st3d.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

- Train with multiple machines
  
  ```shell
  sh scripts/ADA/slurm_train_active_st3d.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \ 
   --cfg_file ./cfgs/ADA/waymo-kitti/pvrcnn/active_st3d.yaml \ 
   --pretrained_model ${PRETRAINED_MODEL}
  ```

&ensp;
&ensp;
## All ADA Results:

We report the cross-dataset adaptation results including Waymo-to-KITTI, nuScenes-to-KITTI, Waymo-to-nuScenes, and Waymo-to-Lyft.

- All LiDAR-based models are trained with 2 NVIDIA A100 GPUs and are available for download.

### ADA Results for Waymo-to-KITTI:

|                                                                                      | training time | Adaptation                  | Car@R40 | download |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- |
| [PV-RCNN](../tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor.yaml)              | ~23h@4 A100   | Source Only                 | 67.95 / 27.65 | -     |
| [PV-RCNN](../tools/cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_01.yaml)              | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 87.12 / 78.03 | [Model-58M](https://drive.google.com/file/d/1zCpZRXQx3j_64HafplLpose4a6gDR6nS/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_05.yaml)              | ~10h@2 A100   | Bi3D (5% annotation budget) | 89.53 / 81.32 | [Model-58M](https://drive.google.com/file/d/1hbso78eIXyYse8Hv1bvz5FLXkCzva7vb/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-kitti/pvrcnn/active_TQS.yaml)                         | ~1.5h@2 A100  | TQS                         | 82.00 / 72.04 | [Model-58M](https://drive.google.com/file/d/12rkTyCTtmQniZSuEcMC8w68f2bx3WjLK/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-kitti/pvrcnn/active_CLUE.yaml)                        | ~1.5h@2 A100  | CLUE                        | 82.13 / 73.14 | [Model-50M](https://drive.google.com/file/d/1kEiaskXkUMryBi7oSynr9PoCVZmzjdry/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-kitti/pvrcnn/active_st3d.yaml)                        | ~10h@2 A100   | Bi3D+ST3D                   | 87.83 / 81.23 | [Model-58M](https://drive.google.com/file/d/1MPL9l1iVCchuhv2wGW6mLqU8tOLJUb-e/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/DA/waymo_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml)                                                                      | ~16h@4 A100   | Source Only                 | 64.87 / 19.90 | -    | 
| [Voxel R-CNN](../tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor_sn_kitti.yaml) | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 88.09 / 79.14 | [Model-72M](https://drive.google.com/file/d/1F9RlK8z-WtOEHN9RIZt9uuzk4p5PGXBw/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_dual_target_05.yaml)         | ~6h@2 A100    | Bi3D (5% annotation budget) | 90.18 / 81.34 | [Model-72M](https://drive.google.com/file/d/1coUt-R9AatKxE_DrWfYw0Y-nBdlDmoBU/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_TQS.yaml)                           | ~1.5h@2 A100  | TQS                         | 78.26 / 67.11 | [Model-72M](https://drive.google.com/file/d/1ByIEVQ9rn8mSXoyE8yY4441LduNNkqB-/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_CLUE.yaml)              | ~1.5h@2 A100  | CLUE                        | 81.93 / 70.89 | [Model-72M](https://drive.google.com/file/d/1wDlmR9rqHna7zQSOb5ktf3bB0S1xVO_e/view?usp=sharing)            |


### ADA Results for nuScenes-to-KITTI:

|                                                                                      | training time | Adaptation                  | Car@R40 | download |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- |
| [PV-RCNN](../tools/cfgs/DA/nusc_kitti/source_only/pvrcnn_old_anchor.yaml)               | ~23h@4 A100   | Source Only                 | 68.15 / 37.17 | [Model-150M](https://drive.google.com/file/d/1RwK3IrA6hESOzjM6X1vBe9v3RENectpW/view?usp=share_link)         |
| [PV-RCNN](../tools/cfgs/ADA/nuscenes-kitti/pvrcnn/active_dual_target_01.yaml)           | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 87.00 / 77.55 | [Model-58M](https://drive.google.com/file/d/1cTDc79zB7Y53CcU7FDvgodoFebHZBlEC/view?usp=share_link)         |
| [PV-RCNN](../tools/cfgs/ADA/nuscenes-kitti/pvrcnn/active_dual_target_05.yaml)           | ~9h@2 A100    | Bi3D (5% annotation budget) | 89.63 / 81.02 | [Model-58M](https://drive.google.com/file/d/1uN9kUljD8APxI32acbVjkzob5v3t2c4f/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/nuscenes-kitti/pvrcnn/active_TQS.yaml)                      | ~1.5h@2 A100  | TQS                         | 84.66 / 75.40 | [Model-58M](https://drive.google.com/file/d/1VCucZI8mQhGSb0_BBPqax4ZvyjAIOPHI/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/nuscenes-kitti/pvrcnn/active_CLUE.yaml)                     | ~1.5h@2 A100  | CLUE                        | 74.77 / 64.43 | [Model-50M](https://drive.google.com/file/d/1YgnI9nrGzQpSRgErwqCLtNgePzzqLypi/view?usp=share_link)         |
| [PV-RCNN](../tools/cfgs/ADA/nuscenes-kitti/pvrcnn/active_st3d.yaml)                     | ~7h@ 2 A100   | Bi3D+ST3D                   | 89.28 / 79.69 | [Model-58M](https://drive.google.com/file/d/1mZZd0LZPJr_cW7U1QI5UaHbZJX_BNrR4/view?usp=share_link)         |
| [Voxel R-CNN](../tools/cfgs/DA/nusc_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml)      | ~16h@4 A100   | Source Only                 | 68.45 / 33.00 | [Model-191M](https://drive.google.com/file/d/1_L7wd8QqYMG5rv9YHzg2JP5s41csGn1C/view?usp=sharing)            | 
| [Voxel R-CNN](../tools/cfgs/ADA/nuscenes-kitti/voxelrcnn/active_dual_target_01.yaml)    | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 87.33 / 77.24 | [Model-72M](https://drive.google.com/file/d/1pl4LQPLvd6rosCxWLtOVQ2qGfJHTwlr0/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/nuscenes-kitti/voxelrcnn/active_dual_target_05.yaml)    | ~5.5h@2 A100  | Bi3D (5% annotation budget) | 87.66 / 80.22 | [Model-72M](https://drive.google.com/file/d/1jAfLkUdSlfJkzj-L0e3AaUBSoyRp94fm/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/nuscenes-kitti/voxelrcnn/active_TQS.yaml)               | ~1.5h@2 A100  | TQS                         | 79.12 / 68.02 | [Model-73M](https://drive.google.com/file/d/1tjJ2f7EVF1cPq4eGjWMrnyNJaUaWywhR/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/nuscenes-kitti/voxelrcnn/active_CLUE.yaml)              | ~1.5h@2 A100  | CLUE                        | 77.98 / 66.02 | [Model-65M](https://drive.google.com/file/d/1XJXV5XVJ0hzi6EEQrkRm7InUz009ufFp/view?usp=sharing)            |


### ADA Results for Waymo-to-nuScenes:

|                                                                                      | training time | Adaptation                  | Car@R40 | download |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- |
| [PV-RCNN](../tools/cfgs/DA/waymo_nusc/source_only/pv_rcnn_plus_feat_3_vehi_full_train.yaml)                     | ~23h@4 A100   | Source Only                 | 31.02 / 21.21 |  -  |
| [PV-RCNN](../tools/cfgs/ADA/waymo-nuscenes/pvrcnn/active_dual_target_01.yaml)           | ~4h@2 A100    | Bi3D (1% annotation budget) | 45.00 / 30.81 | [Model-58M](https://drive.google.com/file/d/1YV18S2Z_qet5Dv7Bj3Rqt2GnZm_Rx92n/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-nuscenes/pvrcnn/active_dual_target_05.yaml)           | ~12h@4 A100   | Bi3D (5% annotation budget) | 48.03 / 32.02 | [Model-58M](https://drive.google.com/file/d/1CCmLJmvcfLSDC6kQ77k4DnRuo7t_WdbJ/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-nuscenes/pvrcnn/active_TQS.yaml)                      | ~4h@2 A100    | TQS                         | 35.47 / 25.00 | [Model-58M](https://drive.google.com/file/d/1nusRyvxSpzNZwOGYP_BQH2HYp-Sn-cui/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-nuscenes/pvrcnn/active_CLUE.yaml)                     | ~3h@2 A100    | CLUE                        | 38.18 / 26.96 | [Model-50M](https://drive.google.com/file/d/1bgP0N2pazsaR9ApBU5R2F7d9WyrkREtu/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/DA/waymo_nusc/voxel_rcnn_feat_3.yaml)              | ~16h@4 A100   | Source Only                 | 29.08 / 19.42 | -      |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-nuscenes/voxelrcnn/active_dual_target_01.yaml)    | ~2.5h@2 A100  | Bi3D (1% annotation budget) | 45.47 / 30.49 | [Model-72M](https://drive.google.com/file/d/1bA_gPTD6V0k_yoOx2hHh8mRiMc_57e_7/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-nuscenes/voxelrcnn/active_dual_target_05.yaml)    | ~4h@4 A100    | Bi3D (5% annotation budget) | 46.78 / 32.14 | [Model-72M](https://drive.google.com/file/d/1N89ymKpJgW1vITsIjP71ZjjO7u8xBIf3/view?usp=sharing)            | 
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-nuscenes/voxelrcnn/active_TQS.yaml)               | ~4h@2 A100    | TQS                         | 36.38 / 24.18 | [Model-72M](https://drive.google.com/file/d/1NfygRS5sHwfTfdqUeYf7emgdJUkLQLL-/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-nuscenes/voxelrcnn/active_CLUE.yaml)              | ~3h@2 A100    | CLUE                        | 37.27 / 25.12 | [Model-65M]()         |
| [SECOND](../tools/cfgs/ADA/waymo-nuscenes/second_iou/second_iou.yaml)                                                                           | ~3h@2 A100    | Bi3D(1%)                    | 46.15 / 26.24 | [Model-54M](https://drive.google.com/file/d/1FjBvGM9E1kgWFcV75ifVb3m3K7VZAnn8/view?usp=sharing)            |


### ADA Results for Waymo-to-Lyft:

|                                                                                      | training time | Adaptation                  | Car@R40 | download |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- |
| [PV-RCNN](../tools/cfgs/ADA/waymo-lyft/pvrcnn/source_only.yaml)                         | ~23h@4 A100   | Source Only                 | 70.10 / 53.11 |  -       |
| [PV-RCNN](../tools/cfgs/ADA/waymo-lyft/pvrcnn/active_dual_target_01.yaml)               | ~7h@2 A100    | Bi3D (1% annotation budget) | 79.07 / 63.74 | [Model-58M](https://drive.google.com/file/d/1LpRTwNYONX4dXXm55G1Yue-IZMG6Xwz4/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-lyft/pvrcnn/active_dual_target_05.yaml)               | ~22h@2 A100   | Bi3D (5% annotation budget) | 80.19 / 66.09 | [Model-58M](https://drive.google.com/file/d/1FZ-foxUUYj0nUaX5NkB_VwTCJqAAOOO6/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-lyft/pvrcnn/active_TQS.yaml)                          | ~7h@2 A100    | TQS                         | 70.87 / 55.25 | [Model-58M](https://drive.google.com/file/d/15mHkrf-aeESB5Vy5YaNtCbInCCBC6mBM/view?usp=sharing)            |
| [PV-RCNN](../tools/cfgs/ADA/waymo-lyft/pvrcnn/active_CLUE.yaml)                         | ~5h@2 A100    | CLUE                        | 75.23 / 62.17 | [Model-50M](https://drive.google.com/file/d/1K0s_wdei9EnRa1hFCXLaesyJN904Sk3j/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-lyft/voxelrcnn/source_only.yaml)                  | ~16h@4 A100   | Source Only                 | 70.52 / 53.48 | -       |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-lyft/voxelrcnn/active_dual_target_01.yaml)        | ~7h@2 A100    | Bi3D (1% annotation budget) | 77.00 / 61.23 | [Model-72M](https://drive.google.com/file/d/1Ka0D_BEiNPuz44-LOD5LjQvRI_Yq3XcI/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-lyft/voxelrcnn/active_dual_target_05.yaml)        | ~19h@2 A100   | Bi3D (5% annotation budget) | 79.15 / 65.26 | [Model-72M](https://drive.google.com/file/d/1oEKS-of6cidD6VZADPrI2c9l3Rf9jlWj/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-lyft/voxelrcnn/active_TQS.yaml)                   | ~8h@2 A100    | TQS                         | 71.11 / 56.28 | [Model-73M](https://drive.google.com/file/d/15FVTe7kQppREBnNVOWqKYyNiUAQ5t_iq/view?usp=sharing)            |
| [Voxel R-CNN](../tools/cfgs/ADA/waymo-lyft/voxelrcnn/active_CLUE.yaml)                  | ~5h@2 A100    | CLUE                        | 75.61 / 59.34 | [Model-65M](https://drive.google.com/file/d/1hgHTXvIJmiKBsYuc6hUFvKLN-SR7glbj/view?usp=sharing)            |
