from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch
from .box_filter import BoxFilter

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_A; mean_b
        mean_A = self.boxfilter(x) / N
        mean_b = self.boxfilter(y) / N

        return (mean_A * x + mean_b).float()

