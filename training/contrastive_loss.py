import torch
import torch.distributed as dist
import torch.nn.functional as F
import diffdist.functional as distops


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


def NT_xent_neg(sim_matrix, temperature=0.5, chunk=3, eps=1e-8, mode="naive", **kwargs):
    '''
        Compute NT_xent loss with additional negatives.
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    if mode == "naive":
        eye = torch.eye(B * chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
    elif mode == "shifted_removed":
        # VERSION 1: Avoid to compute the trans that belongs to the "same" distribution for the newly added data.
        blocking_mask = torch.eye(B * chunk, device=device)
        blocking_mask[:B, 2 * B:3 * B] = blocking_mask[:B, 2 * B:3 * B].fill_diagonal_(1)
        blocking_mask[2 * B:3 * B, :B] = blocking_mask[2 * B:3 * B, :B].fill_diagonal_(1)
        blocking_mask[B:2 * B, 2 * B:3 * B] = blocking_mask[B:2 * B, 2 * B:3 * B].fill_diagonal_(1)
        blocking_mask[2 * B:3 * B, B:2 * B] = blocking_mask[2 * B:3 * B, B:2 * B].fill_diagonal_(1)
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - blocking_mask)
        #################################
    else:
        raise NotImplementedError

    # VERSION 2: only compute the trans that belongs to the "same" distribution
    # blocking_mask = torch.eye(B * chunk, device=device)
    # blocking_mask[:B, 2 * B:3 * B] = 1 - blocking_mask[:B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, :B] = 1 - blocking_mask[2 * B:3 * B, :B].fill_diagonal_(1)
    # blocking_mask[B:2 * B, 2 * B:3 * B] = 1 - blocking_mask[B:2 * B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, B:2 * B] = 1 - blocking_mask[2 * B:3 * B, B:2 * B].fill_diagonal_(1)
    # sim_matrix = torch.exp(sim_matrix / temperature) * (1 - blocking_mask)
    #################################

    # VERSION 3: VERSION 1 + mask out 50% randomly.
    # blocking_mask = torch.eye(B * chunk, device=device)
    # blocking_mask[:B, 2 * B:3 * B] = blocking_mask[:B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, :B] = blocking_mask[2 * B:3 * B, :B].fill_diagonal_(1)
    # blocking_mask[B:2 * B, 2 * B:3 * B] = blocking_mask[B:2 * B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, B:2 * B] = blocking_mask[2 * B:3 * B, B:2 * B].fill_diagonal_(1)
    # sim_matrix = torch.exp(sim_matrix / temperature) * (1 - blocking_mask)
    #################################

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = - torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:2 * B].diag() + sim_matrix[B:2 * B, :B].diag()) / (chunk * B)

    return loss


def NT_xent_neg_exp(sim_matrix, temperature=0.5, chunk=3, eps=1e-8, block_percent=.5):
    '''
        Compute NT_xent loss with additional negatives.
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    # blocking does not boost that much. Even affect some results.
    # blocking_mask = torch.eye(B * chunk, device=device)  # (B', B') to remove diagonal
    # if block_percent != 0:
    #     # Randomly block out some similarities to reduce the number of negative pairs
    #     perm = torch.randperm(B * chunk * B * chunk, device=device)  # (B' * B')
    #     idx = perm[:int(B * chunk * B * chunk * block_percent)]
    #     mask = torch.zeros(B * chunk * B * chunk, device=device)
    #     mask[idx] = 1
    #     mask = mask.reshape(B*chunk, B*chunk).bool()

    #     blocking_mask[mask] = 1

    #     # Keep the positive pairs.
    #     blocking_mask[:B, B:2 * B] = blocking_mask[:B, B:2 * B].fill_diagonal_(0)
    #     blocking_mask[B:2 * B, :B] = blocking_mask[B:2 * B, :B].fill_diagonal_(0)

    # Enforce count shifting transformed views one-by-one.
    # blocking_mask = torch.eye(B * chunk, device=device)
    # blocking_mask[:B, 2 * B:3 * B] = 1 - blocking_mask[:B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, :B] = 1 - blocking_mask[2 * B:3 * B, :B].fill_diagonal_(1)
    # blocking_mask[B:2 * B, 2 * B:3 * B] = 1 - blocking_mask[B:2 * B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, B:2 * B] = 1 - blocking_mask[2 * B:3 * B, B:2 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, 2 * B:3 * B] = 1

    # Avoid to compute the trans that belongs to the "same" distribution.
    # blocking_mask = torch.eye(B * chunk, device=device)
    # blocking_mask[:B, 2 * B:3 * B] = blocking_mask[:B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, :B] = blocking_mask[2 * B:3 * B, :B].fill_diagonal_(1)
    # blocking_mask[B:2 * B, 2 * B:3 * B] = blocking_mask[B:2 * B, 2 * B:3 * B].fill_diagonal_(1)
    # blocking_mask[2 * B:3 * B, B:2 * B] = blocking_mask[2 * B:3 * B, B:2 * B].fill_diagonal_(1)

    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - blocking_mask)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = - torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:2 * B].diag() + sim_matrix[B:2 * B, :B].diag()) / (chunk * B)

    return loss


def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss


def simsiam_loss_batch(p_batch, z_batch, chunk=3):

    B = p_batch.size(0) // chunk  # B = B' / chunk

    z_batch = z_batch.detach()  # stop gradient
    p_batch = F.normalize(p_batch, dim=1)  # l2-normalize 
    z_batch = F.normalize(z_batch, dim=1)  # l2-normalize 

    p_split = torch.split(p_batch, B, dim=0)
    z_split = torch.split(z_batch, B, dim=0)

    losses = []
    for i in range(chunk):
        for j in range(chunk):
            if i == j:
                continue
            losses.append(-(p_split[i] * z_split[j]).sum(dim=1).mean())

    loss = torch.stack(losses).mean()

    return loss
