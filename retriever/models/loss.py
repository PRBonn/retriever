import torch.nn as nn
import torch


class LazytripletLoss(nn.TripletMarginLoss):
    def __init__(self,
                 margin: float = 1.0,
                 p: float = 2.0,
                 swap: bool = False,
                 lazy: bool = True):
        if lazy:
            super().__init__(margin=margin,
                             p=p,
                             swap=swap,
                             reduction='none')
        else:
            super().__init__(margin=margin,
                             p=p,
                             swap=swap,
                             reduction='mean')

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        return super().forward(anchor, positive, negative).max()


class Distance(nn.Module):
    def __init__(self, p: float = 2):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return ((x1-x2)**2+1e-7).sum(dim=-1).sqrt()


class LazyQuadrupletLoss(nn.Module):
    def __init__(self,
                 margin_1: float = 1.0,
                 margin_2: float = 1.0,
                 p: float = 2.0,
                 lazy: bool = True):
        super().__init__()
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.lazy = lazy
        self.dist = Distance(p)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, second_negative: torch.Tensor = None):
        if second_negative is None:
            d1, _ = self.dist(anchor, positive).max(dim=-1)  # hardest positive
            d2, _ = self.dist(anchor, negative).min(dim=-1)  # hardest negative
            # print(d1.shape,d2.shape)
            return torch.clamp(d1 - d2 + self.margin_1, min=0)
        else:
            d1, _ = self.dist(anchor, positive).max(dim=-1)  # hardest positive
            d2, idx = self.dist(anchor, negative).min(
                dim=-1, keepdim=False)  # hardest negative
            if len(idx.shape) > 0: #TODO: avoid somehow for loop (gather, take,...)
                hard_neg = torch.vstack(
                    [negative[i, idx[i], :]for i in range(len(idx))]).unsqueeze(1)
            else:
                hard_neg = negative[idx, :]
            d3 = self.dist(second_negative, hard_neg).squeeze()
            return (torch.clamp(d1 - d2 + self.margin_1, min=0) + torch.clamp(d1 - d3 + self.margin_2, min=0)).mean()


if __name__ == '__main__':

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
    lazy_triplet = LazytripletLoss()
    lazy_quad = LazyQuadrupletLoss()
    anchor = torch.randn(1, 128, requires_grad=True)
    second_neg = torch.randn(1, 128, requires_grad=True)
    positive = torch.randn(1, 128, requires_grad=True)
    negative = torch.randn(18, 128, requires_grad=True)
    output = triplet_loss(anchor, positive, negative)
    output2 = lazy_triplet(anchor, positive, negative)
    output3 = lazy_quad(anchor, positive, negative)
    # output.backward()
    print()
    print('triplet (mean), max, my_max')
    print(output, output2, output3)

    # multiple positives
    print()
    print('triplet + mult_pos, 1,2, my_max')
    positive = torch.randn(2, 128, requires_grad=True)
    output21 = lazy_triplet(anchor, positive[0, :], negative)
    output22 = lazy_triplet(anchor, positive[1, :], negative)
    output3 = lazy_quad(anchor, positive, negative)
    print(output21, output22, "--> max", output3)

    # second negative
    print()
    print('second neg:')
    output4 = lazy_quad(anchor, positive, negative, second_neg)
    print(output4)

    # multiple batches
    print()
    print('batching')
    anchor = torch.randn(3, 1, 128, requires_grad=True)
    second_neg = torch.randn(3, 1, 128, requires_grad=True)
    positive = torch.randn(3, 2, 128, requires_grad=True)
    negative = torch.randn(3, 18, 128, requires_grad=True)
    output3 = lazy_quad(anchor, positive, negative, second_neg)
    print(output3)
    bla = 0
    for i in range(positive.shape[0]):
        bla += lazy_quad(
            anchor[i, :, :], positive[i, :, :], negative[i, :, :], second_neg[i, :, :])
    print(bla/positive.shape[0])
