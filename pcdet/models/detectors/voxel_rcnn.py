from .detector3d_template import Detector3DTemplate
from .detector3d_template_ada import ActiveDetector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from .detector3d_template_multi_db_3 import Detector3DTemplate_M_DB_3
from pcdet.utils import common_utils
from pcdet.config import cfg

class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class VoxelRCNN_M_DB(Detector3DTemplate_M_DB):
    def __init__(self, model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset,
                         dataset_s2=dataset_s2, source_one_name=source_one_name)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name

    def forward(self, batch_dict):

        # Split the Concat dataset batch into batch_1 and batch_2
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)

        batch_s1 = {}
        batch_s2 = {}
   
        len_of_module = len(self.module_list)
        for k, cur_module in enumerate(self.module_list):
            if k < len_of_module-4:
                batch_dict = cur_module(batch_dict)
            
            if k == len_of_module-4 or k == len_of_module-3:
                if len(split_tag_s1) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s2) == batch_dict['batch_size']:
                    continue
                else:
                    if k == len_of_module-4:
                        batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)
                    batch_s1 = cur_module(batch_s1)

            if k == len_of_module-2 or k == len_of_module-1:
                if len(split_tag_s2) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s1) == batch_dict['batch_size']:
                    continue
                else:
                    batch_s2 = cur_module(batch_s2)
            
        if self.training:
            split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
            if len(split_tag_s1) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s1()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            elif len(split_tag_s2) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s2()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
                loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
                ret_dict = {
                    'loss': loss_1 + loss_2
                }
                return ret_dict, tb_dict_1, disp_dict_1
              
        else:
            # NOTE: When peform the inference, only one dataset can be accessed.
            if 'batch_box_preds' in batch_dict.keys():
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
            elif 'batch_box_preds' in batch_s1.keys():
                pred_dicts_s1, recall_dicts_s1 = self.post_processing(batch_s1)
                pred_dicts_s2, recall_dicts_s2 = self.post_processing(batch_s2)
                return pred_dicts_s1, recall_dicts_s1, pred_dicts_s2, recall_dicts_s2

    def get_training_loss_s1(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s2(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

class VoxelRCNN_M_DB_3(Detector3DTemplate_M_DB_3):
    def __init__(self, model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3,
                        dataset=dataset, dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name
        self.source_1 = source_1

    def forward(self, batch_dict):
        batch_s1 = {}
        batch_s2 = {}
        batch_s3 = {}

        if self.training:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-6:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-6 or k == len_of_module-5:
                    # Split the Concat dataset batch into batch_1, batch_2, and batch_3
                    if k == len_of_module-6:
                        split_tag_s1, split_tag_s2_pre = common_utils.split_batch_dict('waymo', batch_dict)
                        batch_s1, batch_s2_pre = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2_pre, batch_dict)
                        split_tag_s2, split_tag_s3 = common_utils.split_batch_dict(self.source_one_name, batch_s2_pre)
                        batch_s2, batch_s3 = common_utils.split_two_batch_dict_gpu(split_tag_s2, split_tag_s3, batch_s2_pre)
                    batch_s1 = cur_module(batch_s1)

                if k == len_of_module-4 or k == len_of_module-3:              
                    batch_s2 = cur_module(batch_s2)

                if k == len_of_module-2 or k == len_of_module-1:
                    batch_s3 = cur_module(batch_s3)
        else:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-6:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-6 or k == len_of_module-5:
                    if self.source_1 == 1:
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue
                if k == len_of_module-4 or k == len_of_module-3:
                    if self.source_1 == 2:         
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

                if k == len_of_module-2 or k == len_of_module-1:
                    if self.source_1 == 3:  
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

        if self.training:
            loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
            loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
            loss_3, tb_dict_3, disp_dict_3 = self.get_training_loss_s3()
            ret_dict = {
                'loss': loss_1 + loss_2 + loss_3
            }
            return ret_dict, tb_dict_1, disp_dict_1
              
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_s1(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s2(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_training_loss_s3(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head_s3.get_loss()
        loss_rcnn, tb_dict = self.roi_head_s3.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

class ActiveDualVoxelRCNN(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        sample_score = []

        for i in range(batch_size):
            for i in range(batch_size):
                frame_score = {
                    'frame_id': frame_id[i],
                    'domainness_evaluate': domainness_evaluate[i].cpu(),
                    'roi_feature': reweight_roi[i],
                    'total_score': domainness_evaluate[i].cpu()
                }
                sample_score.append(frame_score)
            return sample_score


class VoxelRCNN_TQS(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        elif self.training and forward_args.get('mode', 'train_mul_cls'):
            loss, tb_dict, disp_dict = self.get_mul_cls_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_mul_cls_loss(self, mode='train_mul_cls'):
        disp_dict = {}
        loss, loss_mul, tb_dict = self.roi_head.get_mul_cls_loss()
        return loss_mul, tb_dict, disp_dict
    
    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_dict = self.roi_head.committee_evaluate(batch_dict)
        batch_dict = self.roi_head.uncertainty_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        committee_evaluate = batch_dict['committee_score'].cpu()
        uncertainty_evaluate = batch_dict['uncertainty'].cpu()
        roi_score = batch_dict['cls_preds']
        sample_score = []

        for i in range(batch_size):

            frame_score = {
                'frame_id': frame_id[i],
                'committee_evaluate': committee_evaluate[i],
                'uncertainty_evaluate': uncertainty_evaluate[i],
                'domainness_evaluate': domainness_evaluate[i],
                'roi_feature': reweight_roi[i],
                'roi_score': roi_score[i],
                'total_score': committee_evaluate[i] + uncertainty_evaluate[i] + domainness_evaluate[i]
            }
            sample_score.append(frame_score)
        return sample_score

class VoxelRCNN_CLUE(ActiveDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, **forward_args):
        batch_dict['mode'] = forward_args.get('mode', None) if forward_args is not None else None
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and forward_args.get('mode', None) == 'train_discriminator':
            loss = self.discriminator.get_discriminator_loss(batch_dict, source=forward_args['source'])
            return loss
        
        if self.training and forward_args.get('mode', None) == 'train_detector':
            loss, tb_dict, disp_dict = self.get_detector_loss()
        elif self.training and forward_args.get('mode', 'train_mul_cls'):
            loss, tb_dict, disp_dict = self.get_mul_cls_loss()
        
        elif not self.training and forward_args.get('mode', None) == 'active_evaluate':
            batch_dict = self.post_processing(batch_dict)
            sample_score = self.get_evaluate_score(batch_dict, forward_args['domain'])
            return sample_score
        elif not self.training and forward_args.get('mode', None) == None:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_mul_cls_loss(self, mode='train_mul_cls'):
        disp_dict = {}
        loss, mul_loss, tb_dict = self.roi_head.get_mul_cls_loss()
        return mul_loss, tb_dict, disp_dict
    
    def get_detector_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_evaluate_score(self, batch_dict, domain):
        # batch_dict = self.discriminator.domainness_evaluate(batch_dict)
        batch_size = batch_dict['batch_size']
        frame_id = [str(id) for id in batch_dict['frame_id']]
        # domainness_evaluate = batch_dict['domainness_evaluate'].cpu()
        reweight_roi = batch_dict['reweight_roi']
        roi_score = batch_dict['cls_preds']
        sample_score = []

        for i in range(batch_size):
            for i in range(batch_size):
                frame_score = {
                    'frame_id': frame_id[i],
                    # 'domainness_evaluate': domainness_evaluate[i].cpu(),
                    'roi_feature': reweight_roi[i],
                    'roi_score': roi_score[i]
                    # 'total_score': domainness_evaluate[i].cpu()
                }
                sample_score.append(frame_score)
            return sample_score
