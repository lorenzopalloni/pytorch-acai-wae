import torch

from collections import deque
from config import knobs


class Collector:
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.values = deque(maxlen=self.max_len)

    def append(self, x):
        self.values.append(x)

    def __len__(self):
        return len(self.values)

    def mean(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)

    def max(self):
        return max(self.values)

    def min(self):
        return min(self.values)


def reconstruction_loss_func(x: torch.Tensor, y: torch.Tensor):
    return torch.mean(torch.sum(torch.pow(x - y, 2), [1, 2, 3]))


def norm22(x: torch.Tensor):
    return torch.mean(torch.pow(x, 2))


def imq_mmd_func(x, y):
    sigma = knobs['sigma'] ** 2
    bs = x.size(0)
    fbs = float(bs)

    norms_x = x.pow(2).sum(1).unsqueeze(0)
    dotprods_x = torch.mm(x, x.t())
    distances_x = norms_x + norms_x.t() - 2.0 * dotprods_x

    norms_y = y.pow(2).sum(1).unsqueeze(0)
    dotprods_y = torch.mm(y, y.t())
    distances_y = norms_y + norms_y.t() - 2.0 * dotprods_y

    dotprods = torch.mm(y, x.t())
    distances = norms_y + norms_x.t() - 2.0 * dotprods

    stat = 0.0
    Cbase = 2 * knobs['hidden_dim'] * sigma
    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = Cbase * scale
        res1 = C / (C + distances_y)
        res1 += C / (C + distances_x)
        res1 = res1 * (
                torch.ones(bs, bs).to(knobs["device"]) - torch.eye(bs).to(knobs["device"])
        )
        res1 = res1.sum() / (fbs * fbs - fbs)
        res2 = C / (C + distances)
        res2 = res2.sum() * 2.0 / (fbs * fbs)
        stat += res1 - res2
    return stat


def wasserstein_penalty_func(p, q):
    def cov(x: torch.tensor) -> torch.tensor:
        x = x - torch.mean(x, dim=1, keepdim=True)
        return (1. / (x.size(1) - 1)) * x.matmul(x.t())
    mean_p = p.mean(dim=1)
    cov_p = cov(p)
    mean_q = q.mean(dim=1)
    cov_q = cov(q)
    first = torch.sum(torch.pow(mean_p - mean_q, 2))
    second = cov_p.trace()
    third = cov_q.trace()
    fourth = 2 * torch.trace(torch.matmul(cov_p, cov_q).relu().sqrt())
    return first + second + third - fourth
