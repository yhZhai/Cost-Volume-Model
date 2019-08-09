import matplotlib.pyplot as plt
import flowiz as fz
import torch
from cost_volume import *
from cost_volume_model import PreModel
import numpy as np
from PIL import Image
import torchvision
import os
from torch.nn.init import normal_
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    img1 = Image.open('first.png')
    img2 = Image.open('second.png')

    to_tensor = torchvision.transforms.ToTensor()
    img1 = to_tensor(img1).to(device)
    img2 = to_tensor(img2).to(device)
    dp_f = get_displacement_map(img1, img2, delta_w=4, delta_h=4, tau=1)
    dp_b = get_displacement_map(img2, img1, delta_w=4, delta_h=4, tau=1)
    occlusion_model = Occlusion()
    occlusion, dp_rev = occlusion_model(dp_f.unsqueeze(0), dp_b.unsqueeze(0))

    print('max:', occlusion.max().item(), 'min:', occlusion.min().item())
    occlusion = occlusion.squeeze().detach().cpu().numpy()
    occlusion = (((occlusion - occlusion.min()) / (occlusion.max() - occlusion.min())) * 255.9).astype(np.uint8)
    occlusion = Image.fromarray(occlusion)
    occlusion.save('o.png')
    write_flow('f', dp_f, True)
    write_flow('b', dp_b, True)
    write_flow('rev', dp_rev.squeeze(0), True)


def write_flow(fname, dp, del_file=False):
    objectOutput = open('{}.flo'.format(fname), 'wb')

    np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
    np.array([dp.size(2), dp.size(1)], np.int32).tofile(objectOutput)
    np.array(dp.numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)

    objectOutput.close()
    save_flow_file('{}.flo'.format(fname))
    if del_file:
        os.remove('{}.flo'.format(fname))


def save_flow_file(fname: str):
    img = fz.convert_from_file(fname)
    plt.imsave('.'.join(fname.split('.')[:-1] + ['png']), img)


def get_displacement_map(img1, img2, delta_w=10, delta_h=10, tau=0.5):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    shape = img1.shape[2:]
    pre_model = PreModel().to(device)
    t1 = torch.nn.Conv2d(64, 64, 1).to(device)
    t2 = torch.nn.Conv2d(64, 64, 1).to(device)
    normal_(t1.weight, 0, 0.01)
    normal_(t2.weight, 0, 0.01)
    cost_volume = tCostVolume(delta_w=delta_w, delta_h=delta_h).to(device)
    displacement_map = DisplacementMap(delta_w=delta_w, delta_h=delta_h, tau=tau).to(device)

    img1 = pre_model(img1)
    img2 = pre_model(img2)
    img1 = t1(img1)
    img2 = t2(img2)
    cv = cost_volume(img1, img2)
    dp = displacement_map(cv)
    dp = torch.nn.functional.interpolate(dp, size=shape, mode='bilinear', align_corners=False)
    return dp.squeeze(0).cpu().detach()


def read_flow(path):
    TAG_FLOAT = 202021.25
    if not isinstance(path, io.BufferedReader):
        if not isinstance(path, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=path))
        if not os.path.isfile(path):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=path))
        if not path.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=path.split('.')[-1]))

        flo = open(path, 'rb')
    else:
        flo = path

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    nbands = 2
    tmp = np.frombuffer(flo.read(nbands * width * height * 4),
                        np.float32, count=nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    flo.close()

    return flow


if __name__ == '__main__':
    main()
    save_flow_file('out.flo')
    print('finish')
