import os
from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier, eval_ood_detection
from eval import main as eval_ood_main

if 'sup' in P.mode:
    from training.sup import setup
else:
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

mets = ["baseline", "clean_norm", "similar"]
best_met = {met: 0 for met in mets}
best_met.update({"CSI": 0})
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
            P, model, test_loader, ood_test_loader, mets,
            train_loader=train_loader, simclr_aug=simclr_aug
        )
        logger.log(ood)
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()

        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

        _P_prev = deepcopy(P)
        if P.shift_trans_type in ["diffusion_rotation", "blurgaussian", "blurmedian", "blurnone", "rotation"]:
            P.shift_trans_type = "rotation"
        elif P.shift_trans_type in ["diffusion_cutperm", "blurgaussian_cutperm", "blurmedian_cutperm", "blurnone_cutperm", "cutperm"]:
            P.shift_trans_type = "cutperm"
        else:
            raise ValueError
        P.load_path = os.path.join(logger.logdir, "last.model")
        P.print_score = True
        P.mode = "ood_pre"
        P.ood_score = ["CSI"]
        P.ood_samples = 10
        P.resize_factor = 0.54
        P.resize_fix = False
        P.ood_layer = ["simclr", "shift"]
        auc_dict, bests = eval_ood_main(P)
        P = _P_prev

        for met_name in mets + ["CSI"]:
            if met_name == "CSI":
                met_val = sum([auc_dict[k][met_name] for k in auc_dict.keys()]) / len(auc_dict.keys())
            else:
                met_val = sum([ood[k][met_name] for k in ood.keys()]) / len(ood.keys())

            if met_val > best_met[met_name]:
                logger.log(f"Saving best {met_name} AUC checkpoint")
                best_met[met_name] = met_val
                save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir, name=f"best_{met_name}")
                save_linear_checkpoint(linear_optim.state_dict(), logger.logdir, name=f"best_{met_name}")

        logging_str = ('[Epoch %3d] ' % epoch) + " ".join([f"[Best {met} {best_met[met]:5.4f}]" for met in mets + ["CSI"]])
        logger.log(logging_str)

    if epoch % P.error_step == 0 and ('sup' in P.mode):
        error = test_classifier(P, model, test_loader, epoch, logger=logger)

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_error', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))
