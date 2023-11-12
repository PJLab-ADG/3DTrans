
# Getting Started for Point Cloud Pre-training

Please download and preprocess the point cloud datasets according to the [dataset guidance](GETTING_STARTED_DB.md)

### Pre-training Backbone Network using PointContrast on ONCE

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
> Note that you need to set the `--pretrained_model ${PRETRAINED_MODEL}` using the checkpoint obtained in the above-mentioned pre-training processor.
```shell script
sh scripts/dist_train.sh ${NUM_GPUs} \
--cfg_file ${CONFIG_FILE} \
--batch_size ${BATCH_SIZE} \
--pretrained_model ${PRETRAINED_MODEL} 
```
