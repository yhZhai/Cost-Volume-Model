import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CostVolume(nn.Module):
    def __init__(self, delta_w: int, delta_h: int):
        super(CostVolume, self).__init__()
        assert delta_w >= 0 and delta_h >= 0, 'delta_w and delta_h must not be less than 0'
        self.delta_w = delta_w
        self.delta_h = delta_h
        self.zero_pad = nn.ConstantPad2d((delta_w, delta_w, delta_h, delta_h), 0)
        # self.pad = nn.ReplicationPad2d((delta_w, delta_h))

    def forward(self, img1, img2):
        b, c, h, w = img2.shape
        img2 = self.zero_pad(img2)
        output = torch.zeros([b, h, w, 2 * self.delta_h + 1, 2 * self.delta_w + 1]).to(device).float()
        for dh in range(self.delta_h * 2 + 1):
            for dw in range(self.delta_w * 2 + 1):
                output[:, :, :, dh, dw] = cosine_similarity(img1, img2[:, :, dh: dh + h, dw: dw + w], dim=1)

        return output


class DisplacementMap(nn.Module):
    def __init__(self, tau=1):
        super(DisplacementMap, self).__init__()
        self.tau = tau
        self.softmax = nn.Softmax(dim=3)

    def forward(self, cost_volume):
        b = cost_volume.shape[0]
        cost_volume = cost_volume / self.tau
        cost_volume = torch.exp(cost_volume)
        rou_i = cost_volume.sum(dim=3)
        rou_i = self.softmax(rou_i)
        rou_j = cost_volume.sum(dim=4)
        rou_j = self.softmax(rou_j)

        arange = torch.arange(-2, 3).to(device).float().repeat((b,) + cost_volume.size()[1: 3] + (1,))
        v_x = rou_i * arange
        v_x = torch.sum(v_x, dim=3).unsqueeze(1)
        v_y = rou_j * arange
        v_y = torch.sum(v_y, dim=3).unsqueeze(1)
        return torch.cat([v_x, v_y], dim=1)
