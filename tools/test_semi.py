import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import copy

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_semi_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    print("******model for testing",model)
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            break

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ssl_ckpt_dir = output_dir / 'ssl_ckpt'

    eval_output_dir = output_dir / 'eval'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    logger.info('GPU_NAME=%s' % torch.cuda.get_device_name())

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * cfg.OPTIMIZATION.TEST.BATCH_SIZE_PER_GPU))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    batch_size = {
        'pretrain': cfg.OPTIMIZATION.PRETRAIN.BATCH_SIZE_PER_GPU,
        'labeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.LD_BATCH_SIZE_PER_GPU,
        'unlabeled': cfg.OPTIMIZATION.SEMI_SUP_LEARNING.UD_BATCH_SIZE_PER_GPU,
        'test': cfg.OPTIMIZATION.TEST.BATCH_SIZE_PER_GPU,
    }
    # -----------------------create dataloader & network & optimizer---------------------------
    datasets, dataloaders, samplers = build_semi_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test,
        root_path=cfg.DATA_CONFIG.DATA_PATH,
        workers=args.workers,
        logger=logger,
    )

    MODEL_TEACHER = copy.deepcopy(cfg.MODEL)
    teacher_model = build_network(model_cfg=MODEL_TEACHER, num_class=len(cfg.CLASS_NAMES), dataset=datasets['labeled'])
    MODEL_STUDENT = copy.deepcopy(cfg.MODEL)
    student_model = build_network(model_cfg=MODEL_STUDENT, num_class=len(cfg.CLASS_NAMES), dataset=datasets['labeled'])
    teacher_model.set_model_type('teacher')
    student_model.set_model_type('student')

    with torch.no_grad():
        logger.info('**********************Start evaluation for student model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        eval_ssl_dir = output_dir / 'eval' / 'eval_all' / 'eval_with_student_model'
        eval_ssl_dir.mkdir(parents=True, exist_ok=True)
        repeat_eval_ckpt(
            model = student_model,
            test_loader = dataloaders['test'],
            args = args,
            eval_output_dir = eval_ssl_dir,
            logger = logger,
            ckpt_dir = ssl_ckpt_dir / 'student',
            dist_test=dist_test
        )
        logger.info('**********************End evaluation for student model %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


        logger.info('**********************Start evaluation for teacher model %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        eval_ssl_dir = output_dir / 'eval' / 'eval_all' / 'eval_with_teacher_model'
        eval_ssl_dir.mkdir(parents=True, exist_ok=True)
        teacher_model.set_model_type('origin')
        repeat_eval_ckpt(
            model = teacher_model,
            test_loader = dataloaders['test'],
            args = args,
            eval_output_dir = eval_ssl_dir,
            logger = logger,
            ckpt_dir = ssl_ckpt_dir / 'teacher',
            dist_test=dist_test
        )
        logger.info('**********************End evaluation for teacher model %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
