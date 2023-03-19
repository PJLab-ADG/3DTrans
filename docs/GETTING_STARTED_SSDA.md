

# Getting Started & Problem Definition

The purpose of an Semi-supervised Domain Adaptation (SSDA) task is to learn a generalized model or backbone $F$ on a labeled source domain set $s$ and a semi-supervised target domain set $t$. For the target domain, a portion of labeled samples and the remaining unlabeled samples are assumed to be available during the whole adaptation process. 

&ensp;
# Getting Started & Task Challenges
The major difference between SSDA and ADA is that, the ADA requires the model to select the most informative unlabeled samples to perform the annotation process, while the SSDA assmues that some labeled samples from the target domain can be obtained in advance. The key point of SSDA is how to leverage the unlabeled data that may present a significant domain gap with the source domain data.

&ensp;
&ensp;
# Getting Started ## Training & Testing for SSDA setting

Here, we take Waymo-to-nuScenes adaptation as an example.

## Pretraining stage: train the source-only model on the labeled source domain: 

* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml
```

* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple machines
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml
```

* Train other baseline detectors such as Second using multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/second_feat_3_vehi.yaml
```

* Train other baseline detectors such as PV-RCNN++ using multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvplus_feat_3_vehi.yaml
```

## Evaluate the source-pretrained model:
* We report the best model for all epochs on the validation set.

* Test the source-only models using multiple GPUs
```shell script
sh scripts/dist_test.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml \
--ckpt ${CKPT} 
```

* Test the source-only models using multiple machines
```shell script
sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml \
--ckpt ${CKPT}
```

* Test the source-only models of all ckpts using multiple GPUs
```shell script
sh scripts/dist_test.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml \
--eval_all
```

* Test the source-only models of all ckpts using multiple machines
```shell script
sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml \
--eval_all
```

## Fine-tuning stage: fine-tune the model on the labeled target domain:
* You need to set the `--pretrained_model ${PRETRAINED_MODEL}` when finish the pretraining model stage
* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple machines
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_finetune.yaml \ --pretrained_model ${PRETRAINED_MODEL}
```
* Train FEAT=3 (X,Y,Z) without SN (statistical normalization) using multiple GPUs
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_finetune.yaml \
--pretrained_model ${PRETRAINED_MODEL}
```

## Semi-Supervised stage: training the model on the labeled and unlabeled target domain 
* Train FEAT=3 (X,Y,Z) for [SESS: Self-Ensembling Semi-Supervised](https://arxiv.org/abs/1912.11803) using multiple machines
```shell script
sh scripts/SEMI/slurm_train_semi.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_sess.yaml
```
* Train FEAT=3 (X,Y,Z) for [SESS: Self-Ensembling Semi-Supervised](https://arxiv.org/abs/1912.11803) using multiple GPUs
```shell script
sh scripts/SEMI/dist_train_semi.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_sess.yaml
```

* Train FEAT=3 (X,Y,Z) for PSEUDO LABEL using multiple machines
```shell script
sh scripts/SEMI/slurm_train_semi.sh ${PARTITION} ${JOB_NAME} ${NUM_NODES} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_ps.yaml
```
* Train FEAT=3 (X,Y,Z) for PSEUDO LABEL using multiple GPUs
```shell script
sh scripts/SEMI/dist_train_semi.sh ${NUM_GPUs} \
--cfg_file ./cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_ps.yaml
```

## Evaluating the model on the labeled target domain of validation set:
* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard: 
```shell script
python test_semi.py \
--cfg_file ${CONFIG_FILE} \
--batch_size ${BATCH_SIZE}
```

* To test with multiple machines:
```shell script
sh scripts/SEMI/slurm_test_semi.sh ${PARTITION} ${NUM_NODES} \ 
--cfg_file ${CONFIG_FILE} \
--batch_size ${BATCH_SIZE}
```

&ensp;
&ensp;
## All SSDA Results:
We report the cross-dataset adaptation results including Waymo-to-nuScenes, Waymo-to-ONCE.
* All LiDAR-based models are trained with 4 NVIDIA A100 GPUs and are available for download. 
* The domain adaptation time is measured with 4 NVIDIA A100 GPUs and PyTorch 1.8.1.

### SSDA Results for Waymo-to-NuScenes:
* SSDA under the setting of 5% labeled training frames and 99% unlabeled training frames.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [Second](../tools/cfgs/SSDA/waymo_nusc/source_only/second_feat_3_vehi.yaml) | ~11 hours| source-only(Waymo) | 27.85 / 16.43 | - |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_finetune.yaml) | ~0.4 hours| second_5%_FT | 45.95 / 26.98 | [model-61M](https://drive.google.com/file/d/1JIVqpw2cAL8z6wZwoBeJny9-jhFsee_i/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_sess.yaml) | ~1.8 hours| second_5%_SESS | 47.77 / 28.74 | [model-61M](https://drive.google.com/file/d/15kRtg2Cq-cLtMzvm2urENBYw11knjQzA/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_05_ps.yaml) | ~1.7 hours| second_5%_PS | 47.72 / 29.37 | [model-61M](https://drive.google.com/file/d/1MMOEuKyRhymHQwEk8-ow78sXE_n9-iRv/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml) | ~24 hours| source-only(Waymo) | 40.31 / 23.32 | - |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_finetune.yaml) | ~1.0 hours| pvrcnn_5%_FT | 49.58 / 34.86 | [model-150M](https://drive.google.com/file/d/19k8_DGDmwy93Rw9W1nJlGYehUm-nyB1D/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_sess.yaml) | ~5.5 hours| pvrcnn_5%_SESS | 49.92 / 35.28 | [model-150M](https://drive.google.com/file/d/1K8qZkLhAPjUTBzVbcHeh0Hb7To17ojN1/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_05_ps.yaml) | ~5.4 hours| pvrcnn_5%_PS | 49.84 / 35.07 | [model-150M](https://drive.google.com/file/d/1Hh7OQY2thhrxMCRxpr6Si8Utnf-yOvUy/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/source_only/pvplus_feat_3_vehi.yaml) | ~16 hours| source-only(Waymo) | 31.96 / 19.81 | - |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_finetune.yaml) | ~1.2 hours| pvplus_5%_FT | 49.94 / 34.28 | [model-185M](https://drive.google.com/file/d/1VTSic0I2T_k_Y-Tz64biMXDsj4N5vUF4/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_sess.yaml) | ~4.2 hours| pvplus_5%_SESS | 51.14 / 35.25 | [model-185M](https://drive.google.com/file/d/1lONnkK73dTj5CGNzIyssmkHKhsCZaNxS/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_05_ps.yaml) | ~3.6 hours| pvplus_5%_PS | 50.84 / 35.39 | [model-185M](https://drive.google.com/file/d/1wtV3OjkFXMPNHez9X4EPSFQAyBhYKei3/view?usp=share_link) |

* SSDA under the setting of 1% labeled training frames and 99% unlabeled training frames.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [Second](../tools/cfgs/SSDA/waymo_nusc/source_only/second_feat_3_vehi.yaml) | ~11 hours| source-only(Waymo) | 27.85 / 16.43 | - |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_01_finetune.yaml) | ~0.1 hours| second_1%_FT | 41.29 / 23.30 | [model-61M](https://drive.google.com/file/d/1Otad1vDdq6otuJAel5kogaJ37iu0MmH4/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_01_sess.yaml) | ~0.5 hours| second_1%_SESS | 43.25 / 24.59 | [model-61M](https://drive.google.com/file/d/1agxItfjqwlld8ZO6P9Y4prxbLe2u9rdD/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_nusc/second/second_feat_3_vehi_01_ps.yaml) | ~0.5 hours| second_1%_PS | 44.08 / 26.12 | [model-61M](https://drive.google.com/file/d/1VqlksmaJvLSC2B_ULR5NRNMs64z9vIyG/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/source_only/pvrcnn_feat_3_vehi.yaml) | ~24 hours| source-only(Waymo) | 40.31 / 23.32 | - |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_01_finetune.yaml) | ~0.3 hours| pvrcnn_1%_FT | 47.11 / 32.42  | [model-150M](https://drive.google.com/file/d/19SfxS-f341xjonwxXOYl-XgLK7-MEkti/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_01_sess.yaml) | ~1.4 hours| pvrcnn_1%_SESS | 47.43 / 32.75  | [model-150M](https://drive.google.com/file/d/1Epg-UYO4TCelE61_3z-PcUgTup2UECw2/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_nusc/pvrcnn/pvrcnn_feat_3_vehi_01_ps.yaml) | ~1.4 hours| pvrcnn_1%_PS | 47.24 / 32.54 | [model-150M](https://drive.google.com/file/d/15LDn2GJ5hkq3KLiqVgg44isrvtE5I1iQ/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/source_only/pvplus_feat_3_vehi.yaml) | ~16 hours| source-only(Waymo) | 31.96 / 19.81 | - |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_01_finetune.yaml) | ~0.3 hours| pvplus_1%_FT | 47.65 / 31.44 | [model-185M](https://drive.google.com/file/d/1s-U8GaavHc_n3AP4xsP4UgznUYnumeta/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_01_sess.yaml) | ~1.2 hours| pvplus_1%_SESS | 46.95 / 32.60 | [model-185M](https://drive.google.com/file/d/1r4ONa0C-V5yIPnuHJdlflMKF5jmVsHbE/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_nusc/pvplus/pvplus_feat_3_vehi_01_ps.yaml) | ~1.1 hours| pvplus_1%_PS | 47.51 / 33.28 | [model-185M](https://drive.google.com/file/d/1xZ3M0X_7JCLDFafex02v_7brUBHQD3FO/view?usp=share_link) |


&ensp;
### SSDA Results for Waymo-to-KITTI:
* SSDA under the setting of 5% labeled training frames and 95% unlabeled training frames.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [Second](../tools/cfgs/SSDA/waymo_kitti/source_only/second_feat_3_vehi.yaml) | ~11 hours| source-only(Waymo) | 54.77 / 14.57 | - |
| [Second](..//tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_05_finetune.yaml) | ~0.2 hours| second_5%_FT | 79.84 / 60.53 | [model-61M](https://drive.google.com/file/d/1YBetEZQrAMhGdrKGU3w5hmxWLMX0MJQ1/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_05_sess.yaml) | ~0.7 hours| second_5%_SESS | 82.01 / 66.58 | [model-61M](https://drive.google.com/file/d/18LsqVzLhtDC4IKPynfeQnPCVw3-WhwPM/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_05_ps.yaml) | ~0.7 hours| second_5%_PS | 83.08 / 69.80 | [model-61M](https://drive.google.com/file/d/1DmcxIvv31SAYqa3871nKFwOj-SYCreDJ/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml) | ~24 hours| source-only(Waymo) | 67.96 / 27.65 | - |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_05_finetune.yaml) | ~0.3 hours| pvrcnn_5%_FT | 85.88 / 79.49 | [model-150M](https://drive.google.com/file/d/1bHObgAY-dqdMoyMHczniWxPdmRIg3yoQ/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_05_sess.yaml) | ~1.0 hours| pvrcnn_5%_SESS | 88.01 / 79.89 | [model-150M](https://drive.google.com/file/d/1gqgusEi7j_AZbBX03CeO0FXEbRSQxbvm/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_05_ps.yaml) | ~1.0 hours| pvrcnn_5%_PS | 88.64 / 82.08 | [model-150M](https://drive.google.com/file/d/1gWz43ilTxIJqWHCesv6ryUPzWEtaWOsc/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/source_only/pvplus_feat_3_vehi.yaml) | ~21 hours| source-only(Waymo) | 69.86 / 31.24 | - |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_05_finetune.yaml) | ~0.3 hours| pvplus_5%_FT | 91.12 / 82.21 | [model-185M](https://drive.google.com/file/d/10nQUhiixzt_MQUVkFxXT6Wct-KYL1Xlq/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_05_sess.yaml) | ~0.9 hours| pvplus_5%_SESS | 91.01 / 83.06 | [model-185M](https://drive.google.com/file/d/1b6Itvqvn20I6WXqx-AmGXBR63JXlumYz/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_05_ps.yaml) | ~0.9 hours| pvplus_5%_PS | 91.22 / 84.99 | [model-185M](https://drive.google.com/file/d/1RdgVscgyKm0JjuNZGMTQehQeye7nMIPK/view?usp=share_link) |


* SSDA under the setting of 1% labeled training frames and 99% unlabeled training frames.

|                                             | training time | Adaptation | Car@R40   | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [Second](../tools/cfgs/SSDA/waymo_kitti/source_only/second_feat_3_vehi.yaml) | ~11 hours| source-only(Waymo) | 54.77 / 14.57 | - |
| [Second](../tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_01_finetune.yaml) | ~0.1 hours| second_1%_FT | 77.29 / 54.69 | [model-61M](https://drive.google.com/file/d/1M9w2Qa7wm_ysB5pjiLpyaO69ORhMOzh-/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_01_sess.yaml) | ~0.2 hours| second_1%_SESS | 81.53 / 62.41 | [model-61M](https://drive.google.com/file/d/1tQnIukmLGin14WmumWrS7jYy2WmGoO1L/view?usp=share_link) |
| [Second](../tools/cfgs/SSDA/waymo_kitti/second/second_feat_3_vehi_01_ps.yaml) | ~0.2 hours| second_1%_PS | 81.94 / 64.66 | [model-61M](https://drive.google.com/file/d/18UNRc4IVyx0iedZVMU0VoL7anzedK3cB/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/source_only/pvrcnn_feat_3_vehi.yaml) | ~24 hours| source-only(Waymo) | 67.96 / 27.65 | - |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_01_finetune.yaml) | ~0.1 hours| pvrcnn_1%_FT | 86.30 / 76.65 | [model-150M](https://drive.google.com/file/d/1YoXMWr3D2cCJ9nhgJC70wq9u8SORQuzU/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_01_sess.yaml) | ~0.3 hours| pvrcnn_1%_SESS | 87.07 / 79.36  | [model-150M](https://drive.google.com/file/d/1bQrlw3XfAIn-p6wWLUdx983YUMEklv0Z/view?usp=share_link) |
| [PV-RCNN](../tools/cfgs/SSDA/waymo_kitti/pvrcnn/pvrcnn_feat_3_vehi_01_ps.yaml) | ~0.3 hours| pvrcnn_1%_PS | 90.24 / 81.59 | [model-150M](https://drive.google.com/file/d/12IGWyR4cZkTTkX7gFQVcs22fNt9MiArH/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/source_only/pvplus_feat_3_vehi.yaml) | ~21 hours| source-only(Waymo) | 69.86 / 31.24 | - |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_01_finetune.yaml) | ~0.1 hours| pvplus_1%_FT | 89.70 / 78.94 | [model-185M](https://drive.google.com/file/d/1eifNnpfzBwNO3VMCYXLMYW7FDYTgRv3Z/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_01_sess.yaml) | ~0.3 hours| pvplus_1%_SESS | 90.05 / 81.70 | [model-185M](https://drive.google.com/file/d/1BtNlGpigoSOm4dLkKcx3RcJPttFtxiv4/view?usp=share_link) |
| [PV-RCNN++](../tools/cfgs/SSDA/waymo_kitti/pvplus/pvplus_feat_3_vehi_01_ps.yaml) | ~0.3 hours| pvplus_1%_PS | 90.17 / 82.40 | [model-185M](https://drive.google.com/file/d/1MkZ9739bzSm0FWfa5A81SBa2i5XE41QJ/view?usp=share_link) |