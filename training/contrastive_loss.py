import torch
import torch.distributed as dist
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


def NT_xent_neg(sim_matrix, temperature=0.5, chunk=3, eps=1e-8):
    '''
        Compute NT_xent loss with additional negatives.
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    # with torch.no_grad():
    #     mask = torch.ones_like(sim_matrix)
    #     mask[..., - B:, - B:] = 0
    # sim_matrix = sim_matrix * mask  # remove the inter anomaly

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = - torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:2 * B].diag() + sim_matrix[B:2 * B, :B].diag()) / (chunk * B)

    return loss
