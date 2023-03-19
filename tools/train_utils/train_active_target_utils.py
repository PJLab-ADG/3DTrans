import glob
import os
import pickle
import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.utils import active_learning_utils
from pcdet.datasets import build_dataloader_ada
from pcdet.models import load_data_to_gpu


def train_detector(model, model_func, optimizer, lr_scheduler, source_loader, sample_loader, source_loader_iter, sample_loader_iter,
                   dist_train, optim_cfg, rank, total_it_each_epoch, accumulated_iter_detector, tb_log, tbar, leave_pbar=False):
    model.train()
    if dist_train:
        for p in model.module.parameters():
            p.requires_grad = True
        for p in model.module.discriminator.parameters():
            p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True
        for p in model.discriminator.parameters():
            p.requires_grad = False

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train_detector', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    forward_args = {
        'mode': 'train_detector'
    }
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch_src = next(source_loader_iter)
        except StopIteration:
            source_loader_iter = iter(source_loader)
            batch_src = next(source_loader_iter)
            print('new source iter')

        try:
            batch_sample = next(sample_loader_iter)
        except StopIteration:
            sample_loader_iter = iter(sample_loader)
            batch_sample = next(sample_loader_iter)
            print('new sample iter')

        data_timer = time.time()
        cur_data_time = data_timer - end
        lr_scheduler.step(accumulated_iter_detector)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter_detector)

        optimizer.zero_grad()
        loss_sam, tb_dict_sam, disp_dict = model_func(model, batch_sample, **forward_args)

        loss_src, tb_dict_src, disp_dict = model_func(model, batch_src, **forward_args)
        # loss_sam, tb_dict_sam, disp_dict = model_func(model, batch_sample, forward_args)
        loss = loss_src + optim_cfg.SAMPLE_LOSS_SCALE * loss_sam

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer
        
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter_detector += 1
        cur_batch_time = time.time() - end

        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr_detector': cur_lr, 
                'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter_detector))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss_detector', loss, accumulated_iter_detector)
                tb_log.add_scalar('meta_data/learning_rate_detector', cur_lr, accumulated_iter_detector)
                for key, val in tb_dict_src.items():
                    tb_log.add_scalar('train/detector_src' + key, val, accumulated_iter_detector)
                for key, val in tb_dict_sam.items():
                    tb_log.add_scalar('train/detector_sam' + key, val, accumulated_iter_detector)
    if rank == 0:
        pbar.close()
    return accumulated_iter_detector


def train_discriminator(model, optimizer, lr_scheduler, source_loader, target_loader,
                        source_loader_iter, target_loader_iter, dist_train, total_it_each_epoch,
                        accumulated_iter_discriminator, optim_cfg, tb_log, rank, tbar, leave_pbar=False):
    model.train()
    if dist_train:
        for p in model.module.parameters():
            p.requires_grad = False
        for p in model.module.discriminator.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.discriminator.parameters():
            p.requires_grad = True

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train_discriminator', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch_src = next(source_loader_iter)
        except StopIteration:
            source_loader_iter = iter(source_loader)
            batch_src = next(source_loader_iter)
            print('new source iter')

        try:
            batch_tar = next(target_loader_iter)
        except StopIteration:
            target_loader_iter = iter(target_loader)
            batch_tar = next(target_loader_iter)
            print('new target iter')

        data_timer = time.time()
        cur_data_time = data_timer - end
        lr_scheduler.step(accumulated_iter_discriminator)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter_discriminator)

        optimizer.zero_grad()
        load_data_to_gpu(batch_src)
        forward_args = {
            'mode': 'train_discriminator',
            'source': True
        }
        loss_src = model(batch_src, **forward_args)
        load_data_to_gpu(batch_tar)
        forward_args = {
            'mode': 'train_discriminator',
            'source': False
        }
        loss_tar = model(batch_tar, **forward_args)
        loss = (loss_src + loss_tar) / 2

        tb_dict = {
            'discriminator_loss': loss.item()
        }
        disp_dict = {}

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter_discriminator += 1
        cur_batch_time = time.time() - end

        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr_discriminator': cur_lr,
                'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter_discriminator))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss_discriminator', loss, accumulated_iter_discriminator)
                tb_log.add_scalar('meta_data/learning_rate_discriminator', cur_lr, accumulated_iter_discriminator)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter_discriminator)
    if rank == 0:
        pbar.close()
    return accumulated_iter_discriminator


def train_active_model_dual_tar(model, optimizer, source_train_loader, target_train_loader, source_sample_loader, model_func, 
                         lr_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, 
                         sample_epoch, annotation_budget, target_file_path, sample_save_path, cfg, batch_size, workers, dist_train,
                         source_sampler=None, target_sampler=None, source_sample_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                         max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter_detector, accumulated_iter_discriminator = start_iter, start_iter
    target_name = cfg['DATA_CONFIG_TAR']['DATASET']

    oss = True if cfg.DATA_CONFIG_TAR.get('OSS_PATH', None) is not None else False
    target_list = active_learning_utils.get_target_list(target_file_path, oss=oss)
    sample_list = []
    sample_train_loader = None

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(source_train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(source_train_loader.dataset, 'merge_all_iters_to_one_epoch')
            source_train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(source_train_loader) // max(total_epochs, 1)

        dataloader_iter_src = iter(source_train_loader)
        dataloader_iter_tar = iter(target_train_loader)
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if cur_epoch in sample_epoch:
                accumulated_iter_discriminator = train_discriminator(
                    model, 
                    optimizer[1],
                    lr_scheduler[1],
                    source_train_loader,
                    target_train_loader,
                    dataloader_iter_src, 
                    dataloader_iter_tar,
                    dist_train, 
                    len(target_train_loader),
                    accumulated_iter_discriminator, 
                    optim_cfg.DISCRIMINATOR, 
                    tb_log, rank, tbar
                )

            if cur_epoch in sample_epoch:
                frame_score = active_learning_utils.active_evaluate_dual(model, target_train_loader, rank, domain='target')
                sampled_frame_id, _ = active_learning_utils.active_sample_tar(frame_score, budget=annotation_budget, logger=logger)
                sample_list, info_path = active_learning_utils.update_sample_list_dual(
                    sample_list, target_list, sampled_frame_id, cur_epoch, sample_save_path, target_name, rank, domain='target'
                )
                target_list, target_info_path = active_learning_utils.update_target_list(target_list, sampled_frame_id, cur_epoch, sample_save_path, target_name, rank)

                sample_train_set, sample_train_loader, sample_train_sampler = build_dataloader_ada(
                    dataset_cfg=cfg.DATA_CONFIG_SAMPLE,
                    class_names=cfg.DATA_CONFIG_SAMPLE.CLASS_NAMES,
                    batch_size=batch_size,
                    dist=dist_train, workers=workers,
                    logger=logger,
                    training=True,
                    info_path=info_path,
                    merge_all_iters_to_one_epoch=merge_all_iters_to_one_epoch,
                    total_epochs=total_epochs-cur_epoch
                )
                
                target_train_set, target_train_loader, target_train_sampler = build_dataloader_ada(
                    dataset_cfg=cfg.DATA_CONFIG_TAR,
                    class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
                    batch_size=batch_size,
                    dist=dist_train, workers=workers,
                    logger=logger,
                    training=True,
                    info_path=target_info_path,
                    merge_all_iters_to_one_epoch=merge_all_iters_to_one_epoch,
                    total_epochs=total_epochs-cur_epoch
                )

                dataloader_iter_tar = iter(target_train_loader)
                dataloader_iter_sample = iter(sample_train_loader) if sample_train_loader is not None else None
                dataloader_iter_src_sample = iter(source_sample_loader)
                
            accumulated_iter_detector = train_detector(
                model, 
                model_func, 
                optimizer[0],
                lr_scheduler[0],
                source_sample_loader,
                sample_train_loader,
                dataloader_iter_src_sample,
                dataloader_iter_sample,
                dist_train, 
                optim_cfg, 
                rank, 
                len(source_sample_loader), 
                accumulated_iter_detector, 
                tb_log, tbar
            )
            
            # save trained model
            trained_epoch = cur_epoch + 1

            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, epoch=trained_epoch, it=accumulated_iter_detector), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)