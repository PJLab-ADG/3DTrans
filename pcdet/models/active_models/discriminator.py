from xml.dom.minidom import DOMImplementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ActiveDiscriminator(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.fc = nn.Linear(model_cfg['FEATURE_DIM'], 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.fc.weight)

    def get_discriminator_loss(self, batch_dict, source=True):
        domainness = batch_dict['domainness']
        if source:
            discri_loss = F.binary_cross_entropy(domainness, torch.zeros_like(domainness))
        else:
            discri_loss = F.binary_cross_entropy(domainness, torch.ones_like(domainness))
        return discri_loss

    def get_accuracy(self, batch_dict, source=True):
        batch_size = batch_dict['batch_size']
        domainness = batch_dict['domainness']
        zero = torch.zeros_like(domainness)
        one = torch.ones_like(domainness)
        domainness = torch.where(domainness > 0.5, one, domainness)
        domainness = torch.where(domainness <= 0.5, zero, domainness)
        if source:
            acc = (domainness == zero).sum() / batch_size
        else:
            acc = (domainness == one).sum() / batch_size
        return acc

    def domainness_evaluate(self, batch_dict, source=False):
        domainness = batch_dict['domainness'] if source == False else 1 - batch_dict['domainness']
        # domainness_value = 1 / (math.sqrt(2*3.14) * self.model_cfg.SIGMA) * torch.exp(-(domainness - self.model_cfg.MU).pow(2) / 2 * (self.model_cfg.SIGMA ** 2))
        # batch_dict['domainness_evaluate'] = domainness_value
        batch_dict['domainness_evaluate'] = domainness
        return batch_dict

    def forward(self, batch_dict):
        point_feature = batch_dict['point_features']
        point_feature_score = batch_dict['point_cls_scores']
        point_coords = batch_dict['point_coords']
        point_feature = point_feature * point_feature_score.view(-1, 1)
        batch_size = batch_dict['batch_size']
        scene_feature = point_feature.new_zeros((batch_size, point_feature.shape[-1]))
        for i in range(batch_size):
            mask = point_coords[:, 0] == i
            scene_feature[i] = torch.mean(point_feature[mask], dim=0)
        domainness = self.fc(scene_feature)
        domainness = self.sigmoid(domainness)
        batch_dict['domainness'] = domainness
        return batch_dict