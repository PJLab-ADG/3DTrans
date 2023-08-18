[![arXiv](https://img.shields.io/badge/arXiv-2303.06880-b31b1b.svg)](https://arxiv.org/abs/2303.06880)
[![arXiv](https://img.shields.io/badge/arXiv-2303.05886-b31b1b.svg)](https://arxiv.org/abs/2303.05886)
[![arXiv](https://img.shields.io/badge/arXiv-2303.05886-b31b1b.svg)](https://arxiv.org/abs/2306.00612)
[![GitHub issues](https://img.shields.io/github/issues/PJLab-ADG/3DTrans)](https://github.com/PJLab-ADG/3DTrans/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/PJLab-ADG/3DTrans/pulls)


# 3DTrans: Autonomous Driving Transfer Learning Codebase

`3DTrans` is a lightweight, simple, self-contained open-source codebase for exploring the **Autonomous Driving-oriented Transfer Learning Techniques**, which mainly consists of **four** functions at present:
* Unsupervised Domain Adaptation (UDA) for 3D Point Clouds
* Active Domain Adaptation (ADA) for 3D Point Clouds
* Semi-Supervised Domain Adaptation (SSDA) for 3D Point Clouds
* Multi-dateset Domain Fusion (MDF) for 3D Point Clouds

**This project is developed and maintained by Autonomous Driving Group [at] [Shanghai AI Laboratory](https://www.shlab.org.cn/) (ADLab).**

## Overview
- [3DTrans: Autonomous Driving Transfer Learning Codebase](#3dtrans-autonomous-driving-transfer-learning-codebase)
  - [Overview](#overview)
  - [:fire: News :fire:](#fire-news-fire)
    - [:muscle: TODO List :muscle:](#muscle-todo-list-muscle)
  - [3DTrans Framework Introduction](#3dtrans-framework-introduction)
    - [What does `3DTrans` Codebase do?](#what-does-3dtrans-codebase-do)
  - [Installation for 3DTrans](#installation-for-3dtrans)
  - [Getting Started](#getting-started)
  - [Model Zoo:](#model-zoo)
    - [UDA Results:](#uda-results)
    - [ADA Results:](#ada-results)
    - [SSDA Results:](#ssda-results)
    - [MDF Results:](#mdf-results)
  - [Point Cloud Pre-training for Autonomous Driving Task](#point-cloud-pre-training-for-autonomous-driving-task)
  - [Visualization Tools for 3DTrans](#visualization-tools-for-3dtrans)
  - [Acknowledge](#acknowledge)
  - [Related Projects](#related-projects)
  - [Citation](#citation)


&ensp;
## :fire: News :fire:
- [x] We released the AD-PT pre-trained checkpoints, see <a href=./docs/GETTING_STARTED_PRETRAIN.md#once-ckpt>AD-PT pre-trained checkpoints</a> for pre-trained checkpoints (updated on Aug. 2023).
- [x]  Based on `3DTrans`, we achieved significant performance gains on a series of downstream perception benchmarks including Waymo, nuScenes, and KITTI, under different baseline models like PV-RCNN++, SECOND, CenterPoint, PV-RCNN (updated on Jun. 2023).
- [x] Two papers developed using our `3DTrans` were accepted by CVPR-2023: [Uni3D](https://arxiv.org/abs/2303.06880) and [Bi3D](https://arxiv.org/abs/2303.05886). (updated on Mar. 2023).
- [x] Our `3DTrans` supported the Semi-Supervised Domain Adaptation (SSDA) for 3D Object Detection (updated on Nov. 2022).
- [x] Our `3DTrans` supported the Active Domain Adaptation (ADA) of 3D Object Detection for achieving a good trade-off between high performance and annotation cost (updated on Oct. 2022).
- [x] Our `3DTrans` supported several typical transfer learning techniques (such as [TQS](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Transferable_Query_Selection_for_Active_Domain_Adaptation_CVPR_2021_paper.pdf), [CLUE](https://arxiv.org/abs/2010.08666), [SN](https://arxiv.org/abs/2005.08139), [ST3D](https://arxiv.org/abs/2103.05346), [Pseudo-labeling](https://arxiv.org/abs/2103.05346), [SESS](https://arxiv.org/abs/1912.11803), and [Mean-Teacher](https://arxiv.org/abs/1703.01780)) for autonomous driving-related model adaptation and transfer.
- [x] Our `3DTrans` supported the Multi-domain Dataset Fusion (MDF) of 3D Object Detection for enabling the existing 3D models to effectively learn from multiple off-the-shelf 3D datasets (updated on Sep. 2022).
- [x] Our `3DTrans` supported the Unsupervised Domain Adaptation (UDA) of 3D Object Detection for deploying a well-trained source model to an unlabeled target domain (updated on July 2022).
- [x] The self-learning for unlabeled point clouds and new data augmentation operations (such as Object Scaling (ROS) and Size Normalization (SN)) has been supported for `3DTrans`, which can achieve a performance boost based on the dataset prior of [object-size statistics](docs/STATISTICAL_RESULTS.md)

:rocket: We are actively updating this repository currently, and more **cross-dataset fusion solutions** (including domain attention and mixture-of-experts) and more **low-cost data sampling strategy** will be supported by 3DTrans in the furture, which aims to boost the generalization ability and adaptability of the existing state-of-the-art models. :rocket:

We expect this repository will inspire the research of 3D model generalization since it will push the limits of perceptual performance. :tokyo_tower:

### :muscle: TODO List :muscle:

- [ ] For ADA module, need to add the sequence-level data selection policy (to meet the requirement of practical annotation process).
- [x] Provide experimental findings for the AD-related 3D pre-training (**Our ongoing research**, which currently achieves promising pre-training results towards downstream tasks by exploiting large-scale unlabeled data in ONCE dataset using `3DTrans`).


&ensp;
## 3DTrans Framework Introduction

### What does `3DTrans` Codebase do?

- **Rapid Target-domain Adaptation**: `3DTrans` can boost the 3D perception model's adaptability for an unseen domain only using unlabeled data. For example, [ST3D](https://arxiv.org/abs/2103.05346) supported by `3DTrans` repository achieves a new state-of-the-art model transfer performance for many adaptation tasks, further boosting the transferability of detection models. Besides, `3DTrans` develops many new UDA techniques to solve different types of domain shifts (such as LiDAR-induced shifts or object-size-induced shifts), which includes Pre-SN, Post-SN, and range-map downsampling retraining.

- **Annotation-saving Target-domain Transfer**: `3DTrans` can select the most informative subset of an unseen domain and label them at a minimum cost. For example, `3DTrans` develops the [Bi3D](https://arxiv.org/abs/2303.05886), which selects partial-yet-important target data and labels them at a minimum cost, to achieve a good trade-off between high performance and low annotation cost. Besides, `3DTrans` has integrated several typical transfer learning techniques into the 3D object detection pipeline. For example, we integrate the [TQS](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Transferable_Query_Selection_for_Active_Domain_Adaptation_CVPR_2021_paper.pdf), [CLUE](https://arxiv.org/abs/2010.08666), [SN](https://arxiv.org/abs/2005.08139), [ST3D](https://arxiv.org/abs/2103.05346), [Pseudo-labeling](https://arxiv.org/abs/2103.05346), [SESS](https://arxiv.org/abs/1912.11803), and [Mean-Teacher](https://arxiv.org/abs/1703.01780) for supporting autonomous driving-related model transfer.

- **Joint Training on Multiple 3D Datasets**: `3DTrans` can perform the multi-dataset 3D object detection task. For example, `3DTrans` develops the [Uni3D](https://arxiv.org/abs/2303.06880) for multi-dataset 3D object detection, which enables the current 3D baseline models to effectively learn from multiple off-the-shelf 3D datasets, boosting the reusability of 3D data from different autonomous driving manufacturers.

- **Multi-dataset Support**: `3DTrans` provides a unified interface of dataloader, data augmentor, and data processor for multiple public benchmarks, including Waymo, nuScenes, ONCE, Lyft, and KITTI, etc, which is beneficial to study the transferability and generality of 3D perception models among different datasets. Besides, in order to eliminate the domain gaps between different manufacturers and obtain generalizable representations, `3DTrans` has integrated typical unlabeled pre-training techniques for giving a better parameter initialization of the current 3D baseline models. For example, we integrate the [PointContrast](https://arxiv.org/abs/2007.10985) and [SESS](https://arxiv.org/abs/1912.11803) to support point cloud-based pre-training task.

- **Extensibility for Multiple Models**: `3DTrans` makes the baseline model have the ability of cross-domain/dataset safe transfer and multi-dataset joint training. Without making major changes of the code and 3D model structure, a single-dataset 3D baseline model can be successfully adapted to an unseen domain or dataset by using our `3DTrans`.

- `3DTrans` is developed based on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) codebase, which can be easily integrated with the models developing using `OpenPCDet` repository. Thanks for their valuable open-sourcing!

&ensp;
<p align="center">
  <img src="docs/3DTrans_module_relation.png" width="70%">
  <div>The relations of different modules (UDA, ADA, SSDA, MDF, Pre-training) in 3DTrans: The basic model uses UDA/ADA/SSDA (target-oriented adaptation techniques to alleviate inter-domain shift and perform the inference). Then, MDF can use the results of multi-source domains for pre-training to generate a unified dataset. Finally, pre-training task are utilized to learn generalizable representations from multi-source domains, which can effectively boost the performance of basic model.
</div>
</p>


&ensp;
## Installation for 3DTrans

You may refer to [INSTALL.md](docs/INSTALL.md) for the installation of `3DTrans`.

## Getting Started

* Please refer to [Readme for Datasets](docs/GETTING_STARTED_DB.md) to prepare the dataset and convert the data into the 3DTrans format. Besides, 3DTrans supports the reading and writing data from **Ceph Petrel-OSS**, please refer to [Readme for Datasets](docs/GETTING_STARTED_DB.md) for more details.

* Please refer to [Readme for UDA](docs/GETTING_STARTED_UDA.md) for understanding the problem definition of UDA and performing the UDA adaptation process.

* Please refer to [Readme for ADA](docs/GETTING_STARTED_ADA.md) for understanding the problem definition of ADA and performing the ADA adaptation process.

* Please refer to [Readme for SSDA](docs/GETTING_STARTED_SSDA.md) for understanding the problem definition of SSDA and performing the SSDA adaptation process.

* Please refer to [Readme for MDF](docs/GETTING_STARTED_MDF.md) for understanding the problem definition of MDF and performing the MDF joint-training process.


&ensp;
## Model Zoo: 

We could not provide the Waymo-related pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), but you could easily achieve similar performance by training with the corresponding configs.

### UDA Results:

Here, we report the cross-dataset (Waymo-to-KITTI) adaptation results using the BEV/3D AP performance as the evaluation metric. Please refer to [Readme for UDA](docs/GETTING_STARTED_UDA.md) for experimental results of more cross-domain settings.
* All LiDAR-based models are trained with 4 NVIDIA A100 GPUs and are available for download. 
* For Waymo dataset training, we train the model using 20% data.
* The domain adaptation time is measured with 4 NVIDIA A100 GPUs and PyTorch 1.8.1.
* Pre-SN represents that we perform the [SN (statistical normalization)](https://arxiv.org/abs/2005.08139) operation during the pre-training source-only model stage.
* Post-SN represents that we perform the [SN (statistical normalization)](https://arxiv.org/abs/2005.08139) operation during the adaptation stage.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/DA/waymo_kitti/source_only/pointpillar_1x_feat_3_vehi.yaml) |~7.1 hours| Source-only with SN | 74.98 / 49.31 | - | 
| [PointPillar](tools/cfgs/DA/waymo_kitti/pointpillar_1x_pre_SN_feat_3.yaml) |~0.6 hours| Pre-SN | 81.71 / 57.11 | [model-57M](https://drive.google.com/file/d/1tPx8N75sm_zWsZv3FrwtHXeBlorhf9nP/view?usp=share_link) | 
| [PV-RCNN](tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor_sn_kitti.yaml) | ~23 hours| Source-only with SN | 69.92 / 60.17 | - |
| [PV-RCNN](tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml) | ~23 hours| Source-only | 74.42 / 40.35 | - |
| [PV-RCNN](tools/cfgs/DA/waymo_kitti/pvrcnn_pre_SN_feat_3.yaml) | ~3.5 hours| Pre-SN | 84.00 / 74.57 | [model-156M](https://drive.google.com/file/d/1yt1JtBWyBtZjgE22HJUz6L7K6qeWiqM5/view?usp=share_link) |
| [PV-RCNN](tools/cfgs/DA/waymo_kitti/pvrcnn_post_SN_feat_3.yaml) | ~1 hours| Post-SN | 84.94 / 75.20 | [model-156M](https://drive.google.com/file/d/1hd49JZ5amwP2gkblA8IHnH79ITapX2hF/view?usp=share_link) |
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/source_only/voxel_rcnn_sn_kitti.yaml) | ~16 hours| Source-only with SN | 75.83 / 55.50 | - |
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml) | ~16 hours| Source-only | 64.88 / 19.90 | - |
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/voxel_rcnn_pre_SN_feat_3.yaml) | ~2.5 hours| Pre-SN | 82.56 / 67.32 | [model-201M](https://drive.google.com/file/d/1_D7bnECL7bHL_4WOPhxAprHHmxwC8M7U/view?usp=share_link) |
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/voxel_rcnn_post_SN_feat_3.yaml) | ~2.2 hours| Post-SN | 85.44 / 76.78 | [model-201M](https://drive.google.com/file/d/1v0U3Y9K6pe4JaOC5PIECq_wnR77Il-tl/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_sn_kitti.yaml) | ~20 hours| Source-only with SN | 67.22 / 56.50 | - |
| [PV-RCNN++](tools/cfgs/DA/waymo_kitti/source_only/pv_rcnn_plus_feat_3_vehi_full_train.yaml) | ~20 hours| Source-only | 67.68 / 20.82 | - |
| [PV-RCNN++](tools/cfgs/DA/waymo_kitti/pv_rcnn_plus_post_SN_feat_3.yaml) | ~2.2 hours| Post-SN | 86.86 / 79.86 | [model-193M](https://drive.google.com/file/d/1wDNC5kyg8BihV4zEgY2VntA2V_3jeL-5/view?usp=share_link) |


### ADA Results:

Here, we report the Waymo-to-KITTI adaptation results using the BEV/3D AP performance. Please refer to [Readme for ADA](docs/GETTING_STARTED_ADA.md) for experimental results of more cross-domain settings.
* All LiDAR-based models are trained with 4 NVIDIA A100 GPUs and are available for download. 
* For Waymo dataset training, we train the model using 20% data.
* The domain adaptation time is measured with 4 NVIDIA A100 GPUs and PyTorch 1.8.1.

|                                                                                      | training time | Adaptation                  | Car@R40 | download |
| ------------------------------------------------------------------------------------ | ------------- | --------------------------- | ------- | -------- |
| [PV-RCNN](tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor.yaml)              | ~23h@4 A100   | Source Only                 | 67.95 / 27.65 | -    |
| [PV-RCNN](tools/cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_01.yaml)              | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 87.12 / 78.03 | [Model-58M](https://drive.google.com/file/d/1zCpZRXQx3j_64HafplLpose4a6gDR6nS/view?usp=sharing)            |
| [PV-RCNN](tools/cfgs/ADA/waymo-kitti/pvrcnn/active_dual_target_05.yaml)              | ~10h@2 A100   | Bi3D (5% annotation budget) | 89.53 / 81.32 | [Model-58M](https://drive.google.com/file/d/1hbso78eIXyYse8Hv1bvz5FLXkCzva7vb/view?usp=sharing)            |
| [PV-RCNN](tools/cfgs/ADA/waymo-kitti/pvrcnn/active_TQS.yaml)                         | ~1.5h@2 A100  | TQS                         | 82.00 / 72.04 | [Model-58M](https://drive.google.com/file/d/12rkTyCTtmQniZSuEcMC8w68f2bx3WjLK/view?usp=sharing)            |
| [PV-RCNN](tools/cfgs/ADA/waymo-kitti/pvrcnn/active_CLUE.yaml)                        | ~1.5h@2 A100  | CLUE                        | 82.13 / 73.14 | [Model-50M](https://drive.google.com/file/d/1kEiaskXkUMryBi7oSynr9PoCVZmzjdry/view?usp=sharing)            |
| [PV-RCNN](tools/cfgs/ADA/waymo-kitti/pvrcnn/active_st3d.yaml)                        | ~10h@2 A100   | Bi3D+ST3D                   | 87.83 / 81.23 | [Model-58M](https://drive.google.com/file/d/1MPL9l1iVCchuhv2wGW6mLqU8tOLJUb-e/view?usp=sharing)            |
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/source_only/voxel_rcnn_feat_3_vehi.yaml)                                                                      | ~16h@4 A100   | Source Only                 | 64.87 / 19.90 | -    | 
| [Voxel R-CNN](tools/cfgs/DA/waymo_kitti/source_only/pvrcnn_old_anchor_sn_kitti.yaml) | ~1.5h@2 A100  | Bi3D (1% annotation budget) | 88.09 / 79.14 | [Model-72M](https://drive.google.com/file/d/1F9RlK8z-WtOEHN9RIZt9uuzk4p5PGXBw/view?usp=sharing)            |
| [Voxel R-CNN](tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_dual_target_05.yaml)         | ~6h@2 A100    | Bi3D (5% annotation budget) | 90.18 / 81.34 | [Model-72M](https://drive.google.com/file/d/1coUt-R9AatKxE_DrWfYw0Y-nBdlDmoBU/view?usp=sharing)            |
| [Voxel R-CNN](tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_TQS.yaml)                           | ~1.5h@2 A100  | TQS                         | 78.26 / 67.11 | [Model-72M](https://drive.google.com/file/d/1ByIEVQ9rn8mSXoyE8yY4441LduNNkqB-/view?usp=sharing)            |
| [Voxel R-CNN](tools/cfgs/ADA/waymo-kitti/voxelrcnn/active_CLUE.yaml)              | ~1.5h@2 A100  | CLUE                        | 81.93 / 70.89 | [Model-72M](https://drive.google.com/file/d/1wDlmR9rqHna7zQSOb5ktf3bB0S1xVO_e/view?usp=sharing)            |


### SSDA Results:

We report the target domain results on Waymo-to-nuScenes adaptation using the BEV/3D AP performance as the evaluation metric, and Waymo-to-ONCE adaptation using ONCE evaluation metric. Please refer to [Readme for SSDA](docs/GETTING_STARTED_SSDA.md) for experimental results of more cross-domain settings.
* For Waymo-to-nuScenes adaptation, we employ 4 NVIDIA A100 GPUs for model training. 
* The domain adaptation time is measured with 4 NVIDIA A100 GPUs and PyTorch 1.8.1.
* For Waymo dataset training, we train the model using 20% data.
* second_5%_FT denotes that we use 5% nuScenes training data to fine-tune the Second model.
* second_5%_SESS denotes that we utilize the [SESS: Self-Ensembling Semi-Supervised](https://arxiv.org/abs/1912.11803) method to adapt our baseline model.
* second_5%_PS denotes that we fine-tune the source-only model to nuScenes datasets using 5% labeled data, and perform the pseudo-labeling process on the remaining 95% unlabeled nuScenes data.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [Second](tools/cfgs/SSDA/waymo_nusc/source_only/second_feat_3_vehi.yaml) | ~11 hours| source-only(Waymo) | 27.85 / 16.43 | - |
| [Second](tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_finetune.yaml) | ~0.4 hours| second_5%_FT | 45.95 / 26.98 | [model-61M](https://drive.google.com/file/d/1JIVqpw2cAL8z6wZwoBeJny9-jhFsee_i/view?usp=share_link) |
| [Second](tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_sess.yaml) | ~1.8 hours| second_5%_SESS | 47.77 / 28.74 | [model-61M](https://drive.google.com/file/d/15kRtg2Cq-cLtMzvm2urENBYw11knjQzA/view?usp=share_link) |
| [Second](tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_ps.yaml) | ~1.7 hours| second_5%_PS | 47.72 / 29.37 | [model-61M](https://drive.google.com/file/d/1MMOEuKyRhymHQwEk8-ow78sXE_n9-iRv/view?usp=share_link) |
| [PV-RCNN](tools/cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml) | ~24 hours| source-only(Waymo) | 40.31 / 23.32 | - |
| [PV-RCNN](tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_finetune.yaml) | ~1.0 hours| pvrcnn_5%_FT | 49.58 / 34.86 | [model-150M](https://drive.google.com/file/d/19k8_DGDmwy93Rw9W1nJlGYehUm-nyB1D/view?usp=share_link) |
| [PV-RCNN](tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_sess.yaml) | ~5.5 hours| pvrcnn_5%_SESS | 49.92 / 35.28 | [model-150M](https://drive.google.com/file/d/1K8qZkLhAPjUTBzVbcHeh0Hb7To17ojN1/view?usp=share_link) |
| [PV-RCNN](tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_ps.yaml) | ~5.4 hours| pvrcnn_5%_PS | 49.84 / 35.07 | [model-150M](https://drive.google.com/file/d/1Hh7OQY2thhrxMCRxpr6Si8Utnf-yOvUy/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/SSDA/waymo_nusc/source_only/pvplus_feat_3_vehi.yaml) | ~16 hours| source-only(Waymo) | 31.96 / 19.81 | - |
| [PV-RCNN++](tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_finetune.yaml) | ~1.2 hours| pvplus_5%_FT | 49.94 / 34.28 | [model-185M](https://drive.google.com/file/d/1VTSic0I2T_k_Y-Tz64biMXDsj4N5vUF4/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_sess.yaml) | ~4.2 hours| pvplus_5%_SESS | 51.14 / 35.25 | [model-185M](https://drive.google.com/file/d/1lONnkK73dTj5CGNzIyssmkHKhsCZaNxS/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_ps.yaml) | ~3.6 hours| pvplus_5%_PS | 50.84 / 35.39 | [model-185M](https://drive.google.com/file/d/1wtV3OjkFXMPNHez9X4EPSFQAyBhYKei3/view?usp=share_link) |


* For Waymo-to-ONCE adaptation, we employ 8 NVIDIA A100 GPUs for model training. 
* PS denotes that we pseudo-label the unlabeled ONCE and re-train the model on pseudo-labeled data.
* SESS denotes that we utilize the [SESS](https://arxiv.org/abs/1912.11803) method to adapt the baseline.
* For ONCE, the IoU thresholds for evaluation are 0.7, 0.3, 0.5 for Vehicle, Pedestrian, Cyclist. 

|                                             |  Training ONCE Data | Methods | Vehicle@AP  | Pedestrian@AP  | Cyclist@AP  | download | 
|------------------------|---------------------------------:|:----------:|:----------:|:-------:|:-------:|:---------:|
| [Centerpoint](tools/cfgs/once_models/sup_models/centerpoint.yaml) | Labeled (4K) | Train from scracth | 74.93 |  46.21  |  67.36 | [model-96M](https://drive.google.com/file/d/1KxgDaUpph72a18t0i9ceyXrkfvNWRWBE/view?usp=share_link) |
| [Centerpoint_Pede](tools/cfgs/once_models/sup_models/centerpoint_pede_0075.yaml) |  Labeled (4K) |  PS | - |  49.14  |  - | [model-96M](https://drive.google.com/file/d/19-LN7PkkpIMoBIqV8gydghrpkKJo9LS7/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/once_models/sup_models/pv_rcnn_plus_anchor_3CLS.yaml) |  Labeled (4K) | Train from scracth | 79.78 |  35.91  |  63.18 | [model-188M](https://drive.google.com/file/d/187AomgxaRBTFpm3YqJ_UXp2Lg13t9OVs/view?usp=share_link) |
| [PV-RCNN++](tools/cfgs/once_models/semi_learning_models/mt_pv_rcnn_plus_anchor_3CLS_small.yaml) |  Small Dataset (100K) | SESS | 80.02 |   46.24 |  66.41 |[model-188M](https://drive.google.com/file/d/1hEPwnwZVKSmPDE-7XO45dFMoOTanD-n1/view?usp=share_link) |


### MDF Results:

Here, we report the Waymo-and-nuScenes consolidation results. The models are jointly trained on Waymo and nuScenes datasets, and evaluated on Waymo using the mAP/mAHPH LEVEL_2 and nuScenes using the BEV/3D AP. Please refer to [Readme for MDF](docs/GETTING_STARTED_MDF.md) for more results.
* All LiDAR-based models are trained with 8 NVIDIA A100 GPUs and are available for download. 
* The multi-domain dataset fusion (MDF) training time is measured with 8 NVIDIA A100 GPUs and PyTorch 1.8.1.
* For Waymo dataset training, we train the model using 20% training data for saving training time.
* PV-RCNN-nuScenes represents that we train the PV-RCNN model only using nuScenes dataset, and PV-RCNN-DM indicates that we merge the Waymo and nuScenes datasets and train on the merged dataset. Besides, PV-RCNN-DT denotes the domain attention-aware multi-dataset training.


|                Baseline       |          MDF Methods              | Waymo@Vehicle | Waymo@Pedestrian | Waymo@Cyclist   |  nuScenes@Car | nuScenes@Pedestrian | nuScenes@Cyclist   | 
|--------------------------|---------------------------:|:------------------:|:-------------:|:------------:|:------------:|:-------------:|:------------------:|
| [PV-RCNN-nuScenes](./tools/cfgs/MDF/waymo_nusc/only_nusc/pvrcnn_feat_3_SWEEP_10_gt.yaml) | only nuScenes | 35.59 / 35.21 | 3.95 / 2.55 | 0.94 / 0.92 | 57.78 / 41.10 | 24.52 / 18.56 | 10.24 / 8.25 |
| [PV-RCNN-Waymo](./tools/cfgs/MDF/waymo_nusc/only_waymo/pvrcnn_feat_3_3CLS_gt.yaml) | only Waymo | 66.49 / 66.01 | 64.09 / 58.06 | 62.09 / 61.02 | 32.99 / 17.55 | 3.34 / 1.94 |  0.02 / 0.01  |
| [PV-RCNN-DM](./tools/cfgs/MDF/waymo_nusc/multi_db_pvrcnn_feat_3_merged.yaml) | Direct Merging | 57.82 / 57.40 | 48.24 / 42.81 |  54.63 / 53.64  |  48.67 / 30.43 |  12.66 / 8.12 | 1.67 / 1.04 |
| [PV-RCNN-Uni3D](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_uni3d.yaml) | Uni3D | 66.98 / 66.50 | 65.70 / 59.14 | 61.49 / 60.43 | 60.77 / 42.66|  27.44 / 21.85 | 13.50 / 11.87 |
| [PV-RCNN-DT](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_pvrcnn_feat_3_domain_attention.yaml) | Domain Attention | 67.27 / 66.77 | 65.86 / 59.38  |  61.38 / 60.34  | 60.83 / 43.03   |   27.46 / 22.06  |   13.82 / 11.52   |


|                Baseline       |          MDF Methods              | Waymo@Vehicle | Waymo@Pedestrian | Waymo@Cyclist   |  nuScenes@Car | nuScenes@Pedestrian | nuScenes@Cyclist  | 
|------------------------------|-----------:|:---------:|:-------:|:-------:|:----------:|:---------:|:------:|
| [Voxel-RCNN-nuScenes](./tools/cfgs/MDF/waymo_nusc/only_nusc/voxel_rcnn_feat_3_SWEEP_10_gt.yaml) | only nuScenes | 31.89 / 31.65  | 3.74 / 2.57 |2.41 / 2.37 | 53.63 / 39.05 | 22.48 / 17.85 | 10.86 / 9.70  |
| [Voxel-RCNN-Waymo](./tools/cfgs/MDF/waymo_nusc/only_waymo/voxel_rcnn_feat_3_3CLS_gt.yaml) |  only Waymo | 67.05 / 66.41  | 66.75 / 60.83 | 63.13 / 62.15 | 34.10 / 17.31| 2.99 / 1.69  |  0.05 / 0.01   |
| [Voxel-RCNN-DM](./tools/cfgs/MDF/waymo_nusc/multi_db_voxel_rcnn_feat_3_merged.yaml) | Direct Merging | 58.26 / 57.87 |  52.72 / 47.11   |  50.26 / 49.50  |  51.40 / 31.68   |  15.04 / 9.99  |  5.40 / 3.87 |
| [Voxel-RCNN-Uni3D](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_uni3d.yaml) | Uni3D | 66.76 / 66.29  |  66.62 / 60.51  |  63.36 / 62.42  |  60.18 / 42.23 | 30.08 / 24.37   |  14.60 / 12.32  |
| [Voxel-RCNN-DT](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_domain_attention.yaml) | Domain Attention | 66.96 / 66.50 |  68.23 / 62.00  |  62.57 / 61.64   | 60.42 / 42.81  |  30.49 / 24.92  |  15.91 / 13.35 |


|                Baseline       |          MDF Methods              | Waymo@Vehicle | Waymo@Pedestrian | Waymo@Cyclist   |  nuScenes@Car | nuScenes@Pedestrian | nuScenes@Cyclist  | 
|------------------------------|-----------:|:---------:|:-------:|:-------:|:----------:|:-------:|:------:|
| [PV-RCNN++ DM](./tools/cfgs/MDF/waymo_nusc/multi_db_pvplus_feat_3_merged.yaml) | Direct Merging | 63.79 / 63.38  |  55.03 / 49.75  |  59.88 / 58.99  |  50.91 / 31.46  |   17.07 / 12.15   |   3.10 / 2.20   |
| [PV-RCNN++-Uni3D](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_pvplus_feat_3_uni3d.yaml) | Uni3D | 68.55 / 68.08  |  69.83 / 63.60 |  64.90 / 63.91   | 62.51 / 44.16 |  33.82 / 27.18  |  22.48 / 19.30   |
| [PV-RCNN++-DT](./tools/cfgs/MDF/waymo_nusc/waymo_nusc_pvplus_feat_3_domain_attention.yaml) | Domain Attention | 68.51 / 68.05 |  69.81 / 63.58  |  64.39 / 63.43  | 62.33 / 44.16  |  33.44 / 26.94 | 21.64 / 18.52 |


&ensp;
## Point Cloud Pre-training for Autonomous Driving Task

Based on our research progress on the cross-domain adaptation of multiple autonomous driving datasets, we can utilize the **multi-source datasets** for performing the pre-training task. Here, we present several unsupervised and self-supervised pre-training implementations (including [PointContrast](https://arxiv.org/abs/2007.10985)).

- Please refer to [Readme for point cloud pre-training](docs/GETTING_STARTED_PRETRAIN.md) for starting the journey of 3D perception model pre-training.

-  :muscle: :muscle: We are actively exploring the possibility of boosting the **3D pre-training generalization ability**. The corresponding code and pre-training checkpoints are **coming soon** in 3DTrans-v0.2.0.


&ensp;
## Visualization Tools for 3DTrans

- Our `3DTrans` supports the sequence-level visualization function [Quick Sequence Demo](docs/QUICK_SEQUENCE_DEMO.md) to continuously display the prediction results of ground truth of a selected scene.

- **Visualization Demo**: 
  - [Waymo Sequence-level Visualization Demo1](docs/seq_demo_waymo_bev.gif)
  - [Waymo Sequence-level Visualization Demo2](docs/seq_demo_waymo_fp.gif)
  - [nuScenes Sequence-level Visualization Demo](docs/seq_demo_nusc.gif)
  - [ONCE Sequence-level Visualization Demo](docs/seq_demo_once.gif)

&ensp;
## Acknowledge
* Our code is heavily based on [OpenPCDet v0.5.2](https://github.com/open-mmlab/OpenPCDet). Thanks OpenPCDet Development Team for their awesome codebase.

* Our pre-training 3D point cloud task is based on [ONCE Dataset](https://once-for-auto-driving.github.io/). Thanks ONCE Development Team for their inspiring data release.


**Our Papers**: 
- [Uni3D: A Unified Baseline for Multi-dataset 3D Object Detection](https://arxiv.org/abs/2303.06880)<br> 
- [Bi3D: Bi-domain Active Learning for Cross-domain 3D Object Detection](https://arxiv.org/abs/2303.05886)<br>
- [AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset](https://arxiv.org/abs/2306.00612)<br>
- [SUG: Single-dataset Unified Generalization for 3D Point Cloud Classification](https://arxiv.org/abs/2305.09160)<br>


**Our Team**:

- A Team Home for Member Information and Profile, [Project Link](https://bobrown.github.io/Team_3DTrans.github.io/)

## Related Projects
- Welcome to see [SUG](https://github.com/SiyuanHuang95/SUG) for exploring the possibilities of adapting a model to multiple
unseen domains and studying how to leverage the feature’s multi-modal information residing in a single dataset. You can also use [SUG](https://github.com/SiyuanHuang95/SUG) to obtain generalizable features by means of a single dataset, which can obtain good results on unseen domains or datasets.

## Citation
If you find this project useful in your research, please consider citing:
```
@misc{3dtrans2023,
    title={3DTrans: An Open-source Codebase for Exploring Transferable Autonomous Driving Perception Task},
    author={3DTrans Development Team},
    howpublished = {\url{https://github.com/PJLab-ADG/3DTrans}},
    year={2023}
}
```

```
@inproceedings{zhang2023uni3d,
  title={Uni3D: A Unified Baseline for Multi-dataset 3D Object Detection},
  author={Zhang, Bo and Yuan, Jiakang and Shi, Botian and Chen, Tao and Li, Yikang and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9253--9262},
  year={2023}
}
```

```
@inproceedings{yuan2023bi3d,
  title={Bi3D: Bi-domain Active Learning for Cross-domain 3D Object Detection},
  author={Yuan, Jiakang and Zhang, Bo and Yan, Xiangchao and Chen, Tao and Shi, Botian and Li, Yikang and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15599--15608},
  year={2023}
}
```

```
@article{yuan2023AD-PT,
  title={AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset},
  author={Yuan, Jiakang and Zhang, Bo and Yan, Xiangchao and Chen, Tao and Shi, Botian and Li, Yikang and Qiao, Yu},
  journal={arXiv preprint arXiv:2306.00612},
  year={2023}
}
```

```
@article{huang2023sug,
  title={SUG: Single-dataset Unified Generalization for 3D Point Cloud Classification},
  author={Huang, Siyuan and Zhang, Bo and Shi, Botian and Gao, Peng and Li, Yikang and Li, Hongsheng},
  journal={arXiv preprint arXiv:2305.09160},
  year={2023}
}
```