from argparse import ArgumentParser

_P = None


def parse_args(default=False):
    # Make it singleton
    if _P is None:
        return _parse_args(default=default)
    return _P


def _parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['cifar10', 'cifar100', "pneumoniamnist", "breastmnist", "chestmnist", 'imagenet', 'covid', 'kvasir', 'aptos', "breakhis", "retina", "chestxray"], type=str)
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--model', help='Model', type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='simclr', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=[
                            'rotation', 'cutperm', 'none', 'blurgaussian', 'blurmedian', 'blurnone', 'diffusion', 'diffusion_rotation',
                            'diffusion_cutperm', 'blurgaussian_cutperm', 'blurmedian_cutperm', 'blurnone_cutperm'
                        ], type=str)
    parser.add_argument('--diffusion_model_path', default=None, type=str)
    parser.add_argument('--loss_mode', default="shifted_removed", type=str)
    parser.add_argument('--diffusion_scheduler', default='uniform', type=str)
    parser.add_argument('--diff_resolution', help='intermediate product resolution', default=32, type=int)
    parser.add_argument('--contamination_ratio', help='Contaminate training data with anomalies.', default=0., type=float)
    parser.add_argument('--kernel_size', help='blur kernel size',
                        default=7, type=int)
    parser.add_argument("--resize_to_constant", help='resize_to_constant',
                        action='store_true')

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)
    parser.add_argument('--diffusion_offset', help='diffusion_offset',
                        default=.3, type=float)
    parser.add_argument('--diffusion_upper_offset', help='diffusion_upper_offset',
                        default=1., type=float)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--simsiam_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
