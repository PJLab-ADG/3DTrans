
# Getting Started ## Point Cloud Pre-training

Please download and preprocess the point cloud datasets according to the [dataset guidance](GETTING_STARTED.md)

:rocket: :rocket: News: 3DTrans has been verified to effectively improve the precision of pseudo labeling for massive unlabeled data on ONCE dataset!

&ensp;
## Pre-training ONCE using PointContrast 

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
## Pre-training ONCE using Dynamic Updating Pseudo-labeling

We are actively exploring the possibility of boosting the **3D model generalization** by means of ONCE dataset,  The corresponding code is **coming soon** !! :muscle: :muscle: