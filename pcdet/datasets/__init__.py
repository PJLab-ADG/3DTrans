import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import ConcatDataset

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .kitti.kitti_dataset_ada import ActiveKittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .nuscenes.nuscenes_dataset_ada import ActiveNuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .waymo.waymo_dataset_ada import ActiveWaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .lyft.lyft_dataset_ada import ActiveLyftDataset

from .once.once_dataset import ONCEDataset
from .once.once_dataset_ada import ActiveONCEDataset
from .once.once_semi_dataset import ONCEPretrainDataset, ONCELabeledDataset, ONCEUnlabeledDataset, ONCETestDataset, ONCEUnlabeledPairDataset, split_once_semi_data
from .once.once_dataset_pretrain import ONCEDatasetPretrain_ADPT
from .nuscenes.nuscenes_semi_dataset import NuScenesPretrainDataset, NuScenesLabeledDataset, NuScenesUnlabeledDataset, NuScenesTestDataset, split_nuscenes_semi_data
from .kitti.kitti_semi_dataset import KittiPretrainDataset, KittiLabeledDataset, KittiUnlabeledDataset, KittiTestDataset, split_kitti_semi_data


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'ActiveKittiDataset': ActiveKittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'ActiveNuScenesDataset': ActiveNuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'ActiveWaymoDataset': ActiveWaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'ActiveLyftDataset': ActiveLyftDataset,
    'ActiveONCEDataset': ActiveONCEDataset,
    'ONCEDatasetPretrain_ADPT': ONCEDatasetPretrain_ADPT
}

_semi_dataset_dict = {
    'ONCEDataset': {
        'PARTITION_FUNC': split_once_semi_data,
        'PRETRAIN': ONCEPretrainDataset,
        'LABELED': ONCELabeledDataset,
        'UNLABELED': ONCEUnlabeledDataset,
        'UNLABELED_PAIR': ONCEUnlabeledPairDataset,
        'TEST': ONCETestDataset
    },
    'NuScenesDataset': {
        'PARTITION_FUNC': split_nuscenes_semi_data,
        'PRETRAIN': NuScenesPretrainDataset,
        'LABELED': NuScenesLabeledDataset,
        'UNLABELED': NuScenesUnlabeledDataset,
        'TEST': NuScenesTestDataset
    },
    'KittiDataset': {
        'PARTITION_FUNC': split_kitti_semi_data,
        'PRETRAIN': KittiPretrainDataset,
        'LABELED': KittiLabeledDataset,
        'UNLABELED': KittiUnlabeledDataset,
        'TEST': KittiTestDataset
    }
}

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


def build_dataloader_ada(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, info_path=None, merge_all_iters_to_one_epoch=False, total_epochs=0):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
        sample_info_path=info_path,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def build_dataloader_mdf(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, drop_last=True,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def build_semi_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, merge_all_iters_to_one_epoch=False):

    assert merge_all_iters_to_one_epoch is False

    train_infos, test_infos, labeled_infos, unlabeled_infos = _semi_dataset_dict[dataset_cfg.DATASET]['PARTITION_FUNC'](
        dataset_cfg = dataset_cfg,
        info_paths = dataset_cfg.INFO_PATH,
        data_splits = dataset_cfg.DATA_SPLIT,
        root_path = root_path,
        labeled_ratio = dataset_cfg.LABELED_RATIO,
        logger = logger,
    )

    pretrain_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['PRETRAIN'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = train_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset)
    else:
        pretrain_sampler = None

    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=batch_size['pretrain'], pin_memory=True, num_workers=workers,
        shuffle=(pretrain_sampler is None) and True, collate_fn=pretrain_dataset.collate_batch,
        drop_last=False, sampler=pretrain_sampler, timeout=0
    )

    labeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['LABELED'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = labeled_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled_dataset)
    else:
        labeled_sampler = None
    labeled_dataloader = DataLoader(
        labeled_dataset, batch_size=batch_size['labeled'], pin_memory=True, num_workers=workers,
        shuffle=(labeled_sampler is None) and True, collate_fn=labeled_dataset.collate_batch,
        drop_last=False, sampler=labeled_sampler, timeout=0
    )

    unlabeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['UNLABELED'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = unlabeled_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)
    else:
        unlabeled_sampler = None
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=batch_size['unlabeled'], pin_memory=True, num_workers=workers,
        shuffle=(unlabeled_sampler is None) and True, collate_fn=unlabeled_dataset.collate_batch,
        drop_last=False, sampler=unlabeled_sampler, timeout=0
    )

    test_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['TEST'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = test_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        rank, world_size = common_utils.get_dist_info()
        test_sampler = DistributedSampler(test_dataset, world_size, rank, shuffle=False)
    else:
        test_sampler = None
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size['test'], pin_memory=True, num_workers=workers,
        shuffle=(test_sampler is None) and False, collate_fn=test_dataset.collate_batch,
        drop_last=False, sampler=test_sampler, timeout=0
    )

    datasets = {
        'pretrain': pretrain_dataset,
        'labeled': labeled_dataset,
        'unlabeled': unlabeled_dataset,
        'test': test_dataset
    }
    dataloaders = {
        'pretrain': pretrain_dataloader,
        'labeled': labeled_dataloader,
        'unlabeled': unlabeled_dataloader,
        'test': test_dataloader
    }
    samplers = {
        'pretrain': pretrain_sampler,
        'labeled': labeled_sampler,
        'unlabeled': unlabeled_sampler,
        'test': test_sampler
    }

    return datasets, dataloaders, samplers


def build_unsupervised_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, 
                                  logger=None, merge_all_iters_to_one_epoch=False):
    assert merge_all_iters_to_one_epoch is False

    train_infos, test_infos, labeled_infos, unlabeled_infos = _semi_dataset_dict[dataset_cfg.DATASET]['PARTITION_FUNC'](
        dataset_cfg = dataset_cfg,
        info_paths = dataset_cfg.INFO_PATH,
        data_splits = dataset_cfg.DATA_SPLIT,
        root_path = root_path,
        labeled_ratio = dataset_cfg.LABELED_RATIO,
        logger = logger,
    )

    unlabeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['UNLABELED_PAIR'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = unlabeled_infos,
        root_path=root_path,
        logger=logger,
    )

    if dist:
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)
    else:
        unlabeled_sampler = None
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=batch_size['unlabeled'], pin_memory=True, num_workers=workers,
        shuffle=(unlabeled_sampler is None) and True, collate_fn=unlabeled_dataset.collate_batch,
        drop_last=False, sampler=unlabeled_sampler, timeout=0
    )

    test_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['TEST'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = test_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        rank, world_size = common_utils.get_dist_info()
        test_sampler = DistributedSampler(test_dataset, world_size, rank, shuffle=False)
    else:
        test_sampler = None
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size['test'], pin_memory=True, num_workers=workers,
        shuffle=(test_sampler is None) and False, collate_fn=test_dataset.collate_batch,
        drop_last=False, sampler=test_sampler, timeout=0
    )

    datasets = {
        'unlabeled': unlabeled_dataset,
        'test': test_dataset
    }
    dataloaders = {
        'unlabeled': unlabeled_dataloader,
        'test': test_dataloader
    }
    samplers = {
        'unlabeled': unlabeled_sampler,
        'test': test_sampler
    }

    return datasets, dataloaders, samplers

def build_dataloader_multi_db(dataset_cfg_1, dataset_cfg_2, class_names_1, class_names_2, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset_1 = __all__[dataset_cfg_1.DATASET](
        dataset_cfg=dataset_cfg_1,
        class_names=class_names_1,
        root_path=root_path,
        training=training,
        logger=logger,
    )
    dataset_2 = __all__[dataset_cfg_2.DATASET](
        dataset_cfg=dataset_cfg_2,
        class_names=class_names_2,
        root_path=root_path,
        training=training,
        logger=logger,
    )
    concat_dataset = ConcatDataset([dataset_1, dataset_2])
    if merge_all_iters_to_one_epoch:
        assert hasattr(concat_dataset, 'merge_all_iters_to_one_epoch')
        concat_dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(concat_dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(concat_dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        concat_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset_1.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset_1, dataset_2, dataloader, sampler