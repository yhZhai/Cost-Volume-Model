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


class Occlusion(nn.Module):
    def __init__(self):
        super(Occlusion, self).__init__()

    def forward(self, dp_f, dp_b):
        B, C, H, W = dp_f.size()
        ii = torch.arange(0, H).view(-1, 1).repeat(1, W)
        jj = torch.arange(0, W).view(1, -1).repeat(H, 1)
        ii = ii.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        jj = jj.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        ii += dp_f[:, 0, :, :]
        jj += dp_f[:, 1, :, :]
        ii = ii.clamp(0, H - 1)
        jj = jj.clamp(0, W - 1)
        ii = 2 * ii / max(H - 1, 1) - 1
        jj = 2 * jj / max(W - 1, 1) - 1
        dp_rev = nn.functional.grid_sample(dp_b, torch.cat([jj, ii], dim=1).permute(0, 2, 3, 1))

        occlusion = (torch.norm((dp_f + dp_rev), 1, dim=1) ** 2 >= 0.01 * (torch.norm(dp_f, 1, dim=1) ** 2 + torch.norm(dp_rev, 1, dim=1) ** 2) + 0.05)
        return occlusion, dp_rev


class DisplacementMap(nn.Module):
    def __init__(self, delta_w=2, delta_h=2, tau=1):
        super(DisplacementMap, self).__init__()
        self.tau = tau
        self.delta_w = delta_w
        self.delta_h = delta_h
        self.softmax = nn.Softmax(dim=3)

    def forward(self, cost_volume):
        b = cost_volume.shape[0]
        if self.tau != 0:
            cost_volume = cost_volume / self.tau
            cost_volume = torch.exp(cost_volume)
            rou_i = cost_volume.sum(dim=3)
            rou_i = self.softmax(rou_i)
            rou_j = cost_volume.sum(dim=4)
            rou_j = self.softmax(rou_j)
            arange_x = torch.arange(-self.delta_w, self.delta_w + 1).to(device).float().repeat((b,) + cost_volume.size()[1: 3] + (1,))
            arange_y = torch.arange(-self.delta_h, self.delta_h + 1).to(device).float().repeat((b,) + cost_volume.size()[1: 3] + (1,))
            v_x = rou_i * arange_x
            v_x = torch.sum(v_x, dim=3).unsqueeze(1)
            v_y = rou_j * arange_y
            v_y = torch.sum(v_y, dim=3).unsqueeze(1)
            return torch.cat([v_x, v_y], dim=1)
        else:
            max_i, i_idx = torch.max(cost_volume, dim=3)
            max_j, j_idx = torch.max(cost_volume, dim=4)
            mask = cost_volume[:, :, :, self.delta_h, self.delta_w] != torch.max(max_i, dim=3)[0]
            mask = mask.long()
            i_idx = (i_idx[:, :, :, 0] - self.delta_w) * mask
            j_idx = (j_idx[:, :, :, 0] - self.delta_h) * mask
            return torch.cat([i_idx.unsqueeze(1), j_idx.unsqueeze(1)], dim=1).float()
