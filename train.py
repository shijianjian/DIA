import os
from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import eval_ood_detection
from training.unsup import setup


train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

best_auc_baseline = 0
best_auc_baseline_margin = 0

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {}
    kwargs['linear'] = linear
    kwargs['linear_optim'] = linear_optim
    kwargs['simclr_aug'] = simclr_aug

    train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, num_batches_per_epoch=200, **kwargs)

    model.eval()

    if epoch % P.save_step == 0 and P.local_rank == 0:
        ood = eval_ood_detection(
            P, model, test_loader, ood_test_loader, ["baseline", "baseline_marginalized"],
            train_loader=train_loader, simclr_aug=simclr_aug
        )
        logger.log(ood)
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()
        if ood["one_class_1"]["baseline"] > best_auc_baseline:
            logger.log("Saving best baseline AUC checkpoint")
            best_auc_baseline = ood["one_class_1"]["baseline"]
            save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir, name="best_baseline")
            save_linear_checkpoint(linear_optim.state_dict(), logger.logdir, name="best_baseline")
        if ood["one_class_1"]["baseline_marginalized"] > best_auc_baseline_margin:
            logger.log("Saving best baseline_marginalized AUC checkpoint")
            best_auc_baseline_margin = ood["one_class_1"]["baseline_marginalized"]
            save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir, name="best_baseline_marginalized")
            save_linear_checkpoint(linear_optim.state_dict(), logger.logdir, name="best_baseline_marginalized")
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)
        logger.log('[Epoch %3d] [Best baseline %5.4f] [Best baseline marginalized %5.4f]' % (epoch, best_auc_baseline, best_auc_baseline_margin))
        from eval import main as eval_ood_main
        _P_prev = deepcopy(P)
        if P.shift_trans_type in ["diffusion_rotation", "blurgaussian", "blurmedian", "rotation"]:
            P.shift_trans_type = "rotation"
        elif P.shift_trans_type in ["diffusion_cutperm", "blurgaussian_cutperm", "blurmedian_cutperm", "cutperm"]:
            P.shift_trans_type = "cutperm"
        else:
            raise ValueError
        P.load_path = os.path.join(logger.logdir, "last.model")
        P.print_score = True
        P.mode = "ood_pre"
        P.print_score = True
        P.ood_score = ["CSI"]
        P.ood_samples = 10
        P.resize_factor = 0.54
        P.resize_fix = True
        P.ood_layer = ["simclr", "shift"]
        eval_ood_main(P)
        P = _P_prev
