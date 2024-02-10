import torch

from common.eval import read_args
from common.common import parse_args


def main(P):

    model, test_loader, ood_test_loader, train_loader, simclr_aug = read_args(P)

    model.eval()

    if P.mode == 'test_acc':
        from evals import test_classifier
        with torch.no_grad():
            error = test_classifier(P, model, test_loader, 0, logger=None)
        return error

    elif P.mode == 'test_marginalized_acc':
        from evals import test_classifier
        with torch.no_grad():
            error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)
        return error

    elif P.mode in ['ood', 'ood_pre']:
        if P.mode == 'ood':
            from evals import eval_ood_detection
        else:
            from evals.ood_pre import eval_ood_detection
        with torch.no_grad():
            auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                            train_loader=train_loader, simclr_aug=simclr_aug)

        if P.one_class_idx is not None:
            mean_dict = dict()
            for ood_score in P.ood_score:
                mean = 0
                for ood in auroc_dict.keys():
                    mean += auroc_dict[ood][ood_score]
                mean_dict[ood_score] = mean / len(auroc_dict.keys())
            auroc_dict['one_class_mean'] = mean_dict

        bests = []
        for ood in auroc_dict.keys():
            message = ''
            best_auroc = 0
            for ood_score, auroc in auroc_dict[ood].items():
                message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                if auroc > best_auroc:
                    best_auroc = auroc
            message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
            if P.print_score:
                print(message)
            bests.append(best_auroc)

        _bests = map('{:.4f}'.format, bests)
        print('\t'.join(_bests))
        return (auroc_dict, bests)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    P = parse_args()
    main(P)
