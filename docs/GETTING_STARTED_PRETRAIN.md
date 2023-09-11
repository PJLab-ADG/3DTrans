
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