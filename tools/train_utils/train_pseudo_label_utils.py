import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.utils import self_training_utils


def train_detector(model, model_func, optimizer, lr_scheduler, labeled_loader, unlabeled_loader, labeled_loader_iter,
                   unlabeled_loader_iter, dist_train, optim_cfg, rank, total_it_each_epoch, accumulated_iter_detector, tb_log, tbar, leave_pbar=False):
    total_it_each_epoch = len(unlabeled_loader)
    model.train()

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train_detector', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch_labeled = next(labeled_loader_iter)
        except StopIteration:
            labeled_loader_iter = iter(labeled_loader)
            batch_labeled = next(labeled_loader_iter)
            print('new labeled iter')
        
        try:
            batch_unlabeled = next(unlabeled_loader_iter)
        except StopIteration:
            unlabeled_loader_iter = iter(unlabeled_loader)
            batch_unlabeled = next(unlabeled_loader_iter)
            print('new unlabeled iter')

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
        loss_unlabeled, tb_dict_unlabeled, disp_dict = model_func(model, batch_unlabeled)
        loss_labeled, tb_dict_labeled, disp_dict = model_func(model, batch_labeled)
        loss = loss_labeled + loss_unlabeled

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
                for key, val in tb_dict_labeled.items():
                    tb_log.add_scalar('train/detector_labeled' + key, val, accumulated_iter_detector)
                for key, val in tb_dict_unlabeled.items():
                    tb_log.add_scalar('train/detector_unlabeled' + key, val, accumulated_iter_detector)
    if rank == 0:
        pbar.close()
    return accumulated_iter_detector


def train_model(model, optimizer, labeled_train_loader, unlabeled_train_loader, model_func, 
                lr_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, 
                ps_label_dir, cfg, dist_train, labeled_sampler=None, unlabeled_sampler=None, lr_warmup_scheduler=None,
                ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter

    ps_pkl = self_training_utils.check_already_exsit_pseudo_label(ps_label_dir, start_epoch)
    if ps_pkl is not None:
        logger.info('==> Loading pseudo labels from {}'.format(ps_pkl))
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(unlabeled_train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(labeled_train_loader.dataset, 'merge_all_iters_to_one_epoch')
            labeled_train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(labeled_train_loader) // max(total_epochs, 1)

        labeled_loader_iter = iter(labeled_train_loader)
        unlabeled_loader_iter = iter(unlabeled_train_loader)
        for cur_epoch in tbar:
            if labeled_sampler is not None:
                labeled_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if cur_epoch in [0, 5, 10]:
                cfg.DATA_CONFIG.USE_UNLABELED_PSEUDO_LABEL = True
                unlabeled_train_loader.dataset.eval()
                print("***********update pseudo label**********")
                self_training_utils.save_pseudo_label_epoch(
                    model, unlabeled_train_loader, rank, 
                    leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch
                )
                unlabeled_train_loader.dataset.train()
                commu_utils.synchronize()
                
            accumulated_iter = train_detector(
                model, 
                model_func, 
                optimizer,
                cur_scheduler,
                labeled_train_loader,
                unlabeled_train_loader,
                labeled_loader_iter,
                unlabeled_loader_iter,
                dist_train, 
                optim_cfg, 
                rank, 
                total_it_each_epoch, 
                accumulated_iter, 
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
                    checkpoint_state(model, optimizer=optimizer, epoch=trained_epoch, it=accumulated_iter), filename=ckpt_name,
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