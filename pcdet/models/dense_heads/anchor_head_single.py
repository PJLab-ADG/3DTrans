import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
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
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict


class ActiveAnchorHeadSingle1(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
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
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        data_dict['bev_score'] = cls_preds.max(dim=1)[0].view(-1, 1, *cls_preds.shape[2:])
        data_dict['bev_map'] = cls_preds

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict


class AnchorHeadSingle_TQS(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.input_channels = input_channels
        self.margin_scale = self.model_cfg.get('MARGIN_SCALE', None)

        self._init_cls_layers()
        self.conv_box = nn.Conv2d(
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
        self.init_weights()

    def _init_cls_layers(self):
        self.conv_cls = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_cls1 = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_cls2 = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.xavier_normal_(self.conv_cls1.weight)
        nn.init.xavier_uniform_(self.conv_cls2.weight)
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def get_active_loss(self, mode=None):
        cls_loss, tb_dict = self.get_multi_cls_layer_loss()
        cls_loss_1, tb_dict_1 = self.get_multi_cls_layer_loss(head='cls_preds_1')
        tb_dict.update(tb_dict_1)
        cls_loss_2, tb_dict_2 = self.get_multi_cls_layer_loss(head='cls_preds_2')
        tb_dict.update(tb_dict_2)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        if mode is 'train_detector':
            rpn_loss = cls_loss + box_loss + cls_loss_1 + cls_loss_2
        elif mode == 'train_mul_cls':
            rpn_loss = cls_loss_1 + cls_loss_2
        return rpn_loss, tb_dict

    def get_multi_cls_layer_loss(self, head=None):
        head = 'cls_preds' if head is None else head
        cls_preds = self.forward_ret_dict[head]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
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
        loss_name = 'rpn_loss_cls' + head.split('_')[-1] if head is not None else 'rpn_loss_cls'
        tb_dict = {
            loss_name: cls_loss.item()
        }
        return cls_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # multi-classifier
        cls_preds_1 = self.conv_cls1(spatial_features_2d)
        cls_preds_2 = self.conv_cls2(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # multi-classifier
        cls_preds_1 = cls_preds_1.permute(0, 2, 3, 1).contiguous()
        cls_preds_2 = cls_preds_2.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        self.forward_ret_dict['cls_preds_1'] = cls_preds_1
        self.forward_ret_dict['cls_preds_2'] = cls_preds_2

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def committee_evaluate(self, data_dict):
        batch_size = self.forward_ret_dict['cls_preds_1'].shape[0]
        cls_preds_1 = self.forward_ret_dict['cls_preds_1']
        cls_preds_2 = self.forward_ret_dict['cls_preds_2']
        cls_preds_1 = cls_preds_1.view(batch_size, -1, self.num_class)  # (B, num_anchor, num_class)
        cls_preds_2 = cls_preds_2.view(batch_size, -1, self.num_class)  # (B, num_anchor, num_class)
        distances = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            reweight_cls_1 = cls_preds_1[i]
            reweight_cls_2 = cls_preds_2[i]
            dis = (F.sigmoid(reweight_cls_1) - F.sigmoid(reweight_cls_2)).pow(2)  # (num_pos_anchor, num_class)
            dis = dis.mean(dim=-1).mean()
            distances[i] = dis
        self.forward_ret_dict['committee_evaluate'] = distances
        data_dict['committee_evaluate'] = distances
        return data_dict

    def uncertainty_evaluate(self, data_dict):
        batch_size = self.forward_ret_dict['cls_preds_1'].shape[0]
        cls_preds_1 = self.forward_ret_dict['cls_preds_1'].view(batch_size, -1, self.num_class)
        cls_preds_2 = self.forward_ret_dict['cls_preds_2'].view(batch_size, -1, self.num_class)
        uncertainty = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            reweight_cls_1 = cls_preds_1[i].view(-1, 1)
            reweight_cls_2 = cls_preds_2[i].view(-1, 1)
            reweight_cls_1 = F.sigmoid(reweight_cls_1)
            reweight_cls_2 = F.sigmoid(reweight_cls_2)
            uncertainty_cls_1 = torch.min(torch.cat([torch.ones_like(reweight_cls_1) - reweight_cls_1, reweight_cls_1 - torch.zeros_like(reweight_cls_1)], dim=1)).view(-1).mean()
            uncertainty_cls_2 = torch.min(torch.cat([torch.ones_like(reweight_cls_2) - reweight_cls_2, reweight_cls_2 - torch.zeros_like(reweight_cls_2)], dim=1)).view(-1).mean()
            uncertainty[i] = (uncertainty_cls_1 + uncertainty_cls_2) / 2
        data_dict['uncertainty_evaluate'] = uncertainty
        return data_dict
