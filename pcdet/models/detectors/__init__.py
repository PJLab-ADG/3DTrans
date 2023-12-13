from .detector3d_template import Detector3DTemplate
from .detector3d_template_ada import ActiveDetector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn import PVRCNN_M_DB
from .pv_rcnn import PVRCNN_M_DB_3
from .pv_rcnn import SemiPVRCNN
from .pv_rcnn import ActivePVRCNN_DUAL
from .pv_rcnn import PVRCNN_TQS
from .pv_rcnn import PVRCNN_CLUE
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .second_net_iou import ActiveSECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn import VoxelRCNN_M_DB
from .voxel_rcnn import VoxelRCNN_M_DB_3
from .voxel_rcnn import ActiveDualVoxelRCNN
from .voxel_rcnn import VoxelRCNN_CLUE
from .voxel_rcnn import VoxelRCNN_TQS
from .centerpoint import CenterPoint
from .centerpoint import CenterPoint_M_DB
from .centerpoint import SemiCenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .pv_rcnn_plusplus import PVRCNNPlusPlus_M_DB
from .pv_rcnn_plusplus import SemiPVRCNNPlusPlus
from .IASSD import IASSD
from .semi_second import SemiSECOND, SemiSECONDIoU
from .unsupervised_model.pvrcnn_plus_backbone import PVRCNN_PLUS_BACKBONE
from .pv_rcnn_plusplus import PVRCNNPlusPlus_Bi_Pretrain


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Detector3DTemplate_ADA': ActiveDetector3DTemplate,
    'Detector3DTemplate_M_DB': Detector3DTemplate_M_DB,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PVRCNN_M_DB': PVRCNN_M_DB,
    'PVRCNN_M_DB_3': PVRCNN_M_DB_3,
    'SemiPVRCNN': SemiPVRCNN,
    'ActiveDualPVRCNN': ActivePVRCNN_DUAL,
    'PVRCNN_TQS': PVRCNN_TQS,
    'PVRCNN_CLUE': PVRCNN_CLUE,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'ActiveSECONDNetIoU': ActiveSECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'VoxelRCNN_M_DB': VoxelRCNN_M_DB,
    'VoxelRCNN_M_DB_3': VoxelRCNN_M_DB_3,
    'ActiveDualVoxelRCNN': ActiveDualVoxelRCNN,
    'VoxelRCNN_CLUE': VoxelRCNN_CLUE,
    'VoxelRCNN_TQS': VoxelRCNN_TQS,
    'CenterPoint': CenterPoint,
    'CenterPoint_M_DB':CenterPoint_M_DB,
    'SemiCenterPoint': SemiCenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PVRCNNPlusPlus_M_DB': PVRCNNPlusPlus_M_DB,
    'SemiPVRCNNPlusPlus': SemiPVRCNNPlusPlus,
    'IASSD': IASSD,
    'ActiveDualPVRCNN': ActivePVRCNN_DUAL,
    'SemiSECOND': SemiSECOND,
    'SemiSECONDIoU': SemiSECONDIoU,
    'PVRCNN_PLUS_BACKBONE': PVRCNN_PLUS_BACKBONE,
    'PVRCNNPlusPlus_Bi_Pretrain': PVRCNNPlusPlus_Bi_Pretrain
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

# def build_detector_multi_db_v2(model_cfg, num_class, dataset):
#     model = __all__[model_cfg.NAME](
#         model_cfg=model_cfg, num_class=num_class, dataset=dataset
#     )

#     return model

def build_detector_multi_db(model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset, 
        dataset_s2=dataset_s2, source_one_name=source_one_name
    )

    return model

def build_detector_multi_db_3(model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3, dataset=dataset, 
        dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1
    )

    return model