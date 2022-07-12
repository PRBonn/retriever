import os
import torch

CONFIG_DIR = dir_path = os.path.dirname(os.path.realpath(__file__))+'/../config/'


def knn(q_pts, s_pts, k, cosine_sim=False):
    if cosine_sim:
        sim = torch.einsum('...in,...jn->...ij', q_pts, s_pts)
        _, neighb_inds = torch.topk(sim, k, dim=-1, largest=True)
        return neighb_inds
    else:
        dist = ((q_pts.unsqueeze(-2) - s_pts.unsqueeze(-3))**2).sum(-1)
        _, neighb_inds = torch.topk(dist, k, dim=-1, largest=False)
        return neighb_inds
