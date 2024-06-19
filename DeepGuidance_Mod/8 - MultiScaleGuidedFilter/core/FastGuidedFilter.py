import torch
from torch import nn
import torch.nn.functional as F
from .box_filter import BoxFilter

class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y, "Batch sizes do not match"
        assert c_x == 1 or c_x == c_y, "Channels do not match or not single channel"
        assert h_x == h_y and w_x == w_y, "Spatial dimensions do not match"
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1, "Input is too small for the given radius"

        # Calculate mean and variance
        N = self.boxfilter(torch.ones((1, 1, h_x, w_x), dtype=x.dtype, device=x.device))

        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N

        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        output = mean_A * x + mean_b
        return output.float()

class MultiScaleGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8, scales=[1, 2, 4]):
        super(MultiScaleGuidedFilter, self).__init__()
        self.scales = scales
        self.guided_filters = nn.ModuleList([FastGuidedFilter(r // scale, eps) for scale in scales])

    def forward(self, x, y):
        outputs = []
        for scale, guided_filter in zip(self.scales, self.guided_filters):
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                y_scaled = F.interpolate(y, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
                y_scaled = y

            output = F.interpolate(guided_filter(x_scaled, y_scaled), size=x.shape[-2:], mode='bilinear', align_corners=False)
            outputs.append(output)
        
        # Average the outputs from different scales
        final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return final_output