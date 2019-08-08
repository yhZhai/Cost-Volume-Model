from collections import OrderedDict

import torch

# in1 = torch.randn(100, 128)
# in2 = torch.randn(100, 128)
# out = torch.nn.functional.cosine_similarity(in1, in2)
# print(out)

a: dict = torch.load('ucf101_single_cv_checkpoint_modify.pth.tar')
state_dict = a['state_dict']

ans: OrderedDict = OrderedDict()

for key, value in state_dict.items():
    if key.split('.')[0] == 'module':
        ans['.'.join(key.split('.')[1:])] = value
    else:
        ans[key] = value


torch.save(ans, 'state.pth')
