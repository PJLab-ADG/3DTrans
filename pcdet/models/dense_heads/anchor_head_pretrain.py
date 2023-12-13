import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .anchor_head_template import AnchorHeadTemplate
from ...utils import common_utils
from ...ops.iou3d_nms import iou3d_nms_utils


def random_world_flip(box_preds, params, reverse = False):
    if reverse:
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
    else:
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
    return box_preds

def random_world_rotation(box_preds, params, reverse = False):
    if reverse:
        noise_rotation = -params
    else:
        noise_rotation = params

    angle = torch.tensor([noise_rotation]).to(box_preds.device)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(1)
    ones = angle.new_ones(1)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(3, 3).float()
    box_preds[:, :3] = torch.matmul(box_preds[:, :3], rot_matrix)
    box_preds[:, 6] += noise_rotation
    return box_preds

def random_world_scaling(box_preds, params, reverse = False):
    if reverse:
        noise_scale = 1.0/params
    else:
        noise_scale = params

    box_preds[:, :6] *= noise_scale
    return box_preds

@torch.no_grad()
def reverse_transform(teacher_boxes, teacher_dict, student_dict):
    augmentation_functions = {
        'random_world_flip': random_world_flip,
        'random_world_rotation': random_world_rotation,
        'random_world_scaling': random_world_scaling
    }
    for bs_idx, box_preds in enumerate(teacher_boxes):
        teacher_aug_list = teacher_dict['augmentation_list'][bs_idx]
        student_aug_list = student_dict['augmentation_list'][bs_idx]
        teacher_aug_param = teacher_dict['augmentation_params'][bs_idx]
        student_aug_param = student_dict['augmentation_params'][bs_idx]

        teacher_aug_list = teacher_aug_list[::-1]
        for key in teacher_aug_list:
            aug_params = teacher_aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse = True)

        for key in student_aug_list:
            aug_params = student_aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse = False)
        teacher_boxes[bs_idx] = box_preds
    return teacher_boxes

class AnchorHeadSinglePretrain(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=True
        )

        self.model_cfg = model_cfg

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_cls_1 = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box_1 = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_1 = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls_1 = None

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv_cls_1.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box_1.weight, mean=0, std=0.001)
        

    def split_batch_dict(self, batch_dict):
        batch_tag_1 = []
        batch_tag_2 = []
        # TODO: add data_flag to batch
        for k in range(batch_dict['batch_size']):
            if 'batch_1' in batch_dict['data_flag'][k]:
                batch_tag_1.append(k)
            else:
                batch_tag_2.append(k)
            
        return batch_tag_1, batch_tag_2

    def forward(self, data_dict):
        self.forward_ret_dict_1 = {}
        self.forward_ret_dict_2 = {}
        batch_tag_1, batch_tag_2 = self.split_batch_dict(data_dict)
        data_dict_1, data_dict_2 = common_utils.split_two_batch_dict_gpu(batch_tag_1, batch_tag_2, data_dict)
        # if 'augmentation_list' in data_dict_1.keys():
        #     print('data_dict_1 has augmentation')
        #     print('data_dict_2 has augmentation')
        spatial_features_2d_1 = data_dict_1['spatial_features_2d']
        spatial_features_2d_2 = data_dict_2['spatial_features_2d']

        cls_preds_1 = self.conv_cls(spatial_features_2d_1)
        box_preds_1 = self.conv_box(spatial_features_2d_1)

        cls_preds_2 = self.conv_cls_1(spatial_features_2d_2)
        box_preds_2 = self.conv_box_1(spatial_features_2d_2)

        cls_preds_1 = cls_preds_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds_1 = box_preds_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        cls_preds_2 = cls_preds_2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds_2 = box_preds_2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict_1['cls_preds'] = cls_preds_1
        self.forward_ret_dict_1['box_preds'] = box_preds_1

        self.forward_ret_dict_2['cls_preds'] = cls_preds_2
        self.forward_ret_dict_2['box_preds'] = box_preds_2

        spatial_size = cls_preds_1.shape[0:3]
        # need to debug
        # [2, 188, 188, 6, 3] -> [2, 188, 188, 6]
        select_box_score_1, select_box_inds_1 = cls_preds_1.view(-1, self.num_class).max(dim=-1)
        # [2, 188, 188, 6] -> [2, 188, 188]
        cls_score_1, cls_inds_1 = select_box_score_1.view(-1, self.num_anchors_per_location).max(dim=-1)
        # [2, 188 * 188] -> [2, 256]
        cls_score_1 = cls_score_1.view(spatial_size[0], -1)
        cls_score_select_1, cls_select_index_1 = torch.sort(cls_score_1.view(cls_score_1.shape[0], -1), descending=True, dim=-1)
        cls_select_index_1 = cls_select_index_1[:, :self.model_cfg.NUM_SELECT]
        cls_score_select_1 = cls_score_select_1[:, :self.model_cfg.NUM_SELECT]

        select_box_score_2, select_box_inds_2 = cls_preds_2.view(-1, self.num_class).max(dim=-1)
        cls_score_2, cls_inds_2 = select_box_score_2.view(-1,self.num_anchors_per_location).max(dim=-1)
        cls_score_2 = cls_score_2.view(spatial_size[0], -1)
        cls_score_select_2, cls_select_index_2 = torch.sort(cls_score_2.view(cls_score_2.shape[0], -1), descending=True, dim=-1)
        cls_select_index_2 = cls_select_index_2[:, :self.model_cfg.NUM_SELECT]
        cls_score_select_2 = cls_score_select_2[:, :self.model_cfg.NUM_SELECT]

        if self.conv_dir_cls is not None:
            dir_cls_preds_1 = self.conv_dir_cls(spatial_features_2d_1)
            dir_cls_preds_1 = dir_cls_preds_1.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict_1['dir_cls_preds'] = dir_cls_preds_1
        else:
            dir_cls_preds_1 = None

        if self.conv_dir_cls_1 is not None:
            dir_cls_preds_2 = self.conv_dir_cls_1(spatial_features_2d_2)
            dir_cls_preds_2 = dir_cls_preds_2.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict_2['dir_cls_preds'] = dir_cls_preds_2
        else:
            dir_cls_preds_2 = None  

        # get box info
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds_1, batch_box_preds_1 = self.generate_predicted_boxes(
                batch_size=data_dict_1['batch_size'],
                cls_preds=cls_preds_1, box_preds=box_preds_1, dir_cls_preds=dir_cls_preds_1
            )
            data_dict_1['batch_cls_preds'] = batch_cls_preds_1
            data_dict_1['batch_box_preds'] = batch_box_preds_1
            data_dict_1['cls_preds_normalized'] = False

            batch_cls_preds_2, batch_box_preds_2 = self.generate_predicted_boxes(
                batch_size=data_dict_2['batch_size'],
                cls_preds=cls_preds_2, box_preds=box_preds_2, dir_cls_preds=dir_cls_preds_2
            )
            data_dict_2['batch_cls_preds'] = batch_cls_preds_2
            data_dict_2['batch_box_preds'] = batch_box_preds_2
            data_dict_2['cls_preds_normalized'] = False

        batch_box_preds_1 = batch_box_preds_1.view(spatial_size[0], -1, self.num_anchors_per_location, self.box_coder.code_size)
        batch_box_preds_2 = batch_box_preds_2.view(spatial_size[0], -1, self.num_anchors_per_location, self.box_coder.code_size)
        
        cls_inds_1 = cls_inds_1.view(spatial_size[0], -1)
        cls_inds_2 = cls_inds_2.view(spatial_size[0], -1)

        # spatial_size = box_preds_1.shape[0:3]
        select_boxes_1 = box_preds_1.new_zeros(*cls_select_index_1.shape, self.box_coder.code_size)
        select_boxes_2 = box_preds_2.new_zeros(*cls_select_index_2.shape, self.box_coder.code_size)
        select_boxes_id_1 = box_preds_1.new_zeros(*cls_select_index_1.shape)
        select_boxes_id_2 = box_preds_2.new_zeros(*cls_select_index_2.shape)
        select_features_1 = box_preds_1.new_zeros(*cls_select_index_1.shape, spatial_features_2d_1.shape[1])
        select_features_2 = box_preds_2.new_zeros(*cls_select_index_2.shape, spatial_features_2d_2.shape[1])

        for i  in range(spatial_size[0]):
            cur_select_1, cur_select_2 = cls_select_index_1[i], cls_select_index_2[i]
            cur_pred_box_1, cur_pred_box_2 = batch_box_preds_1[i], batch_box_preds_2[i]
            cur_pred_box_1 = cur_pred_box_1[cur_select_1]
            cur_pred_box_2 = cur_pred_box_2[cur_select_2]
            cur_select_box_inds_1 = cls_inds_1[i][cur_select_1]
            cur_select_box_inds_2 = cls_inds_2[i][cur_select_2]
            select_features_1[i] = spatial_features_2d_1.permute(0, 2, 3, 1).contiguous().view(-1, spatial_features_2d_1.shape[1])[cur_select_1]
            select_features_2[i] = spatial_features_2d_2.permute(0, 2, 3, 1).contiguous().view(-1, spatial_features_2d_2.shape[1])[cur_select_2]
            select_boxes_id_1[i] = cur_select_1 * self.num_anchors_per_location + cur_select_box_inds_1
            select_boxes_id_2[i] = cur_select_2 * self.num_anchors_per_location + cur_select_box_inds_2

            for j in range(select_boxes_1.shape[1]):
                select_boxes_1[i][j] = cur_pred_box_1[j][cur_select_box_inds_1[j]]
                select_boxes_2[i][j] = cur_pred_box_2[j][cur_select_box_inds_2[j]]

        select_boxes_id_1 = select_boxes_id_1.long()
        select_boxes_id_2 = select_boxes_id_2.long()

        select_box_1_reverse = reverse_transform(select_boxes_1, data_dict_1, data_dict_2)

        # TODO Feature level consistency: how to confirm?
        consist_mask_list_1, consist_mask_list_2 = self.get_consist_mask_1(select_box_1_reverse, select_boxes_2, consist_thr=self.model_cfg.CONSIST_THR)
        # print('consist_mask: ', consist_mask_list_1[0].shape)
        consist_feature_list_1, consist_feature_list_2 = [], []
        consist_boxes_id_1, consist_boxes_id_2 = [], []
        for i in range(len(consist_mask_list_1)):
            consist_feature_list_1.append(select_features_1[i][consist_mask_list_1[i]])
            consist_feature_list_2.append(select_features_2[i][consist_mask_list_2[i]])
            consist_boxes_id_1.append(select_boxes_id_1[i][consist_mask_list_1[i]])
            consist_boxes_id_2.append(select_boxes_id_2[i][consist_mask_list_2[i]])

            # print('consist_feature shape: ', select_features_1[i][consist_mask_list_1[i]].shape)

        self.forward_ret_dict_1['consist_feature'] = consist_feature_list_1
        self.forward_ret_dict_2['consist_feature'] = consist_feature_list_2

        if self.training and not self.model_cfg.get('ASSIGN_TARGETS_WITH_MASK', False):
            targets_dict = self.assign_targets(
                gt_boxes=data_dict_1['gt_boxes']
            )
            self.forward_ret_dict_1.update(targets_dict)

            targets_dict = self.assign_targets(
                gt_boxes=data_dict_2['gt_boxes']
            )
            self.forward_ret_dict_2.update(targets_dict)
        
        elif self.training and self.model_cfg.get('ASSIGN_TARGETS_WITH_MASK', False):
            targets_dict = self.assign_targets_with_mask(
                gt_boxes=data_dict_1['gt_boxes'], mask_inds=consist_boxes_id_1
            )
            self.forward_ret_dict_1.update(targets_dict)

            targets_dict = self.assign_targets_with_mask(
                gt_boxes=data_dict_2['gt_boxes'], mask_inds=consist_boxes_id_2
            )
            self.forward_ret_dict_2.update(targets_dict)
        
        data_dict = common_utils.merge_two_batch_dict_gpu(data_dict_1, data_dict_2)

        return data_dict
    
    def get_consist_mask(self, pred_box_1, pred_box_2, max_num=256, consist_thr=0.3):
        batch_size = pred_box_1.shape[0]
        consist_mask_list_1, consist_mask_list_2 = [], []
        for k in range(batch_size):
            cur_box_1, cur_box_2 = pred_box_1[k], pred_box_2[k]
            overlap = iou3d_nms_utils.boxes_iou3d_gpu(cur_box_1[:, 0:7], cur_box_2[:, 0:7])
            max_overlap, inds = overlap.max(dim=1)
            sorted_overlap, sort_inds = torch.sort(max_overlap, descending=True)
            if len((sorted_overlap > consist_thr).nonzero()) > max_num:
                consist_mask_1 = sort_inds[:max_num]
                consist_mask_2 = inds[consist_mask_1]
            else:
                consist_mask_1 = sort_inds[sorted_overlap > consist_thr]
                consist_mask_2 = inds[consist_mask_1]
            consist_mask_list_1.append(consist_mask_1)
            consist_mask_list_2.append(consist_mask_2)
        return consist_mask_list_1, consist_mask_list_2

    def get_consist_mask_1(self, pred_box_1, pred_box_2, consist_thr=0.25):
        consist_mask_list_1, consist_mask_list_2 = [], []
        pred_box_1_center = pred_box_1[:, :, :3]
        pred_box_2_center = pred_box_2[:, :, :3]
        for i in range(pred_box_1_center.shape[0]):
            cur_center_1 = pred_box_1_center[i].unsqueeze(dim=1).repeat(1, self.model_cfg.NUM_SELECT, 1)
            cur_center_2 = pred_box_2_center[i].unsqueeze(dim=0).repeat(self.model_cfg.NUM_SELECT, 1, 1)
            center_distance = ((cur_center_1 - cur_center_2)**2).sum(dim=-1)
            distance, dis_inds = center_distance.min(dim=-1)
            consist_mask_1 = (distance < consist_thr).nonzero().squeeze()
            # print('consist_mask: ', consist_mask_1.shape)
            consist_mask_2 = dis_inds[consist_mask_1]
            consist_mask_list_1.append(consist_mask_1)
            consist_mask_list_2.append(consist_mask_2)
        return consist_mask_list_1, consist_mask_list_2
        
    def get_consistent_loss(self):
        total_loss = 0.
        feature_1 = self.forward_ret_dict_1['consist_feature']
        feature_2 = self.forward_ret_dict_2['consist_feature']
        mean_feature_num = 0
        for i in range(len(feature_1)):
            if len(feature_1[i]) == 0:
                continue
            loss = torch.pow(feature_1[i] - feature_2[i], 2).mean(dim=-1).mean(dim=-1)
            # loss = torch.pow(feature_1[i] - feature_2[i], 2).sum(dim=-1).mean(dim=-1)
            total_loss += loss
            mean_feature_num += len(feature_1[i])
        total_loss = total_loss / len(feature_1)
        mean_feature_num /= len(feature_1)
        if isinstance(total_loss, float):
            tb_dict = {'mean_consist_num': mean_feature_num}
            return total_loss, tb_dict
        else:
            tb_dict = {'consist_loss': total_loss.item(), 'mean_consist_num': mean_feature_num}
        tb_dict = {'consist_loss': total_loss.item()}
        return total_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        consist_loss, tb_dict_consist = self.get_consistent_loss()
        tb_dict.update(tb_dict_consist)
        rpn_loss = cls_loss + box_loss  + consist_loss * self.model_cfg.get('CONSIST_LOSS_FACTOR', 1.)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
    
    def get_cls_layer_loss(self):
        cls_loss_total = 0.
        for forward_ret_dict in [self.forward_ret_dict_1, self.forward_ret_dict_2]:
            cls_preds = forward_ret_dict['cls_preds']
            box_cls_labels = forward_ret_dict['box_cls_labels']
            batch_size = int(cls_preds.shape[0])
            cared = box_cls_labels >= 0  # [N, num_anchors]
            positives = box_cls_labels > 0
            positives_known = (box_cls_labels >0) & (box_cls_labels < (len(self.class_names) + 1))
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives_known.float()
            # reg_weights = positives.float()
            if self.num_class == 1:
                # class agnostic
                box_cls_labels[positives] = 1

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size

            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            cls_loss_total += cls_loss
        
        cls_loss_total /= 2

        tb_dict = {
            'rpn_loss_cls': cls_loss_total.item()
        }
        return cls_loss_total, tb_dict
    
    def get_box_reg_layer_loss(self):
        box_loss_total= 0.
        for forward_ret_dict in [self.forward_ret_dict_1, self.forward_ret_dict_2]:
            box_preds = forward_ret_dict['box_preds']
            box_dir_cls_preds = forward_ret_dict.get('dir_cls_preds', None)
            box_reg_targets = forward_ret_dict['box_reg_targets']
            box_cls_labels = forward_ret_dict['box_cls_labels']
            batch_size = int(box_preds.shape[0])

            positives = (box_cls_labels >0) & (box_cls_labels < (len(self.class_names) + 1))
            # positives = box_cls_labels > 0
            reg_weights = positives.float()
            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            if isinstance(self.anchors, list):
                if self.use_multihead:
                    anchors = torch.cat(
                        [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                        self.anchors], dim=0)
                else:
                    anchors = torch.cat(self.anchors, dim=-3)
            else:
                anchors = self.anchors
            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            box_preds = box_preds.view(batch_size, -1,
                                    box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                    box_preds.shape[-1])
            # sin(a - b) = sinacosb-cosasinb
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_loss = loc_loss
            tb_dict = {
                'rpn_loss_loc': loc_loss.item()
            }

            if box_dir_cls_preds is not None:
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )

                dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                box_loss += dir_loss
                tb_dict['rpn_loss_dir'] = dir_loss.item()
            box_loss_total += box_loss
        box_loss_total /= 2
        return box_loss_total, tb_dict
