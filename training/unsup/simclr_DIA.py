import time

import torch.optim
import torch.nn as nn


import models.transform_layers as TL
from models.classifier import get_simclr_augmentation
from training.contrastive_loss import get_similarity_matrix, NT_xent_neg, simsiam_loss_batch
from utils.utils import AverageMeter, normalize

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

rand_quant = TL.RandomizedQuantizationAugModule(8)


def diffuser_sampler_by_epoch(epoch, T_max=30, mean_value=20, if_reverse=False):
    step = (mean_value - 2) / T_max * 2
    epoch = epoch % T_max
    if epoch < T_max / 2:
        alpha = mean_value
        beta = step * epoch + 2
    else:
        beta = mean_value
        alpha = step * (T_max - epoch - 1) + 2

    if if_reverse:
        return torch.distributions.beta.Beta(beta, alpha)
    return torch.distributions.beta.Beta(alpha, beta)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None, eval_fn=None, num_batches_per_epoch=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_shift > 1, P.K_shift

    # _resize_scale_factor = min(1 + epoch, 8)
    # assert P.resize_factor * _resize_scale_factor < 1.
    # simclr_aug = get_simclr_augmentation(
    #     P, image_size=P.image_size,
    #     resize_scale_factor=_resize_scale_factor,
    #     rand_quant=False,
    # ).to(device)

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()
    if P.model.endswith("simsiam"):
        losses['simsiam'] = AverageMeter()

    if P.diffusion_scheduler == "beta_by_epoch":
        P.shift_trans._step_sampler = diffuser_sampler_by_epoch(epoch)
    elif P.diffusion_scheduler == "beta_by_epoch_reversed":
        P.shift_trans._step_sampler = diffuser_sampler_by_epoch(epoch, if_reverse=True)
    elif P.diffusion_scheduler == "beta_skewed":
        P.shift_trans._step_sampler = torch.distributions.Beta(1, 1.7)
    elif P.diffusion_scheduler == "beta_skewed_reversed":
        P.shift_trans._step_sampler = torch.distributions.Beta(1.7, 1.)
    elif P.diffusion_scheduler == "uniform":
        P.shift_trans._step_sampler = torch.distributions.Uniform(0, 1)
    else:
        raise RuntimeError

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        # Additional negative samples, which shall make negative pairs to all other samples
        images3 = torch.cat([P.shift_trans(hflip(images), k, diffusion=True, diffusion_offset=P.diffusion_offset, diffusion_upper_offset=P.diffusion_upper_offset) for k in range(P.K_shift)])

        images1 = torch.cat([P.shift_trans(images1, k, diffusion=False) for k in range(P.K_shift)])
        images2 = torch.cat([P.shift_trans(images2, k, diffusion=False) for k in range(P.K_shift)])
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(3)
        # shift_labels = torch.cat([shift_labels.repeat(2), shift_labels + P.K_Shift])

        images_pair = torch.cat([images1, images2, images3], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  # transform

        # SimSiam-like
        if P.model.endswith("simsiam"):
            _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True, projector=True, predictor=True)
            loss_simsiam = simsiam_loss_batch(
                # outputs_aux['predictor'][images1.shape[0] // 3 * 2:],
                # outputs_aux['projector'][images1.shape[0] // 3 * 2:], chunk=2
                outputs_aux['predictor'], outputs_aux['projector'], chunk=3
            ) * P.simsiam_lambda
        else:
            _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize dim 1
        sim_matrix = get_similarity_matrix(simclr, chunk=3, multi_gpu=P.multi_gpu)

        loss_sim = NT_xent_neg(sim_matrix, temperature=0.5, mode=P.loss_mode) * P.sim_lambda

        # loss_shift = criterion(outputs_aux['shift'][:-P.K_shift * batch_size], shift_labels[:-P.K_shift * batch_size])
        loss_shift = criterion(outputs_aux['shift'], shift_labels)

        ### total loss ###
        if P.model.endswith("simsiam"):
            loss = loss_sim + loss_shift + loss_simsiam
        else:
            loss = loss_sim + loss_shift

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][P.K_shift * batch_size: (P.K_shift + 1) * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)
        if P.model.endswith("simsiam"):
            losses['simsiam'].update(loss_simsiam.item(), batch_size)

        if count % 50 == 0 or (num_batches_per_epoch is not None and count >= num_batches_per_epoch):
            # TODO: eval
            if P.model.endswith("simsiam"):
                log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                    '[LossC %f] [LossSim %f] [LossSimSiam %f] [LossShift %f]' %
                    (epoch, count, batch_time.value, data_time.value, lr,
                    losses['cls'].value, losses['sim'].value, losses['simsiam'].value, losses['shift'].value))
            else:
                log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                    '[LossC %f] [LossSim %f] [LossShift %f]' %
                    (epoch, count, batch_time.value, data_time.value, lr,
                    losses['cls'].value, losses['sim'].value, losses['shift'].value))

            if n >= num_batches_per_epoch:
                break

    if P.model.endswith("simsiam"):
        log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossSimSiam %f] [LossShift %f]' %
            (batch_time.average, data_time.average,
            losses['cls'].average, losses['sim'].average, losses['simsiam'].average, losses['shift'].average))
    else:
        log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
            (batch_time.average, data_time.average,
            losses['cls'].average, losses['sim'].average, losses['shift'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        if P.model.endswith("simsiam"):
            logger.scalar_summary('train/loss_simsiam', losses['simsiam'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
