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
