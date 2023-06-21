import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from utils import *
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc


def test_detect(data_loader, net, get_pbb, save_dir, config, n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = os.path.basename(data_loader.dataset.filenames[i_name])
        shortname = os.path.splitext(name)[0]
        if shortname.endswith('_clean'):
            shortname = shortname[:-len('_clean')]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        print((data.size()))
        splitlist = list(range(0, len(data) + 1, max(n_gpu, 1)))
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        if n_gpu == 0:
            device_wrap = lambda x: x.to(dtype=torch.float32, device='cpu')
        else:
            device_wrap = lambda x: x.cuda()

        with torch.no_grad():
            for i in range(len(splitlist) - 1):
                input = device_wrap(
                    Variable(data[splitlist[i]:splitlist[i + 1]],
                             volatile=True))
                inputcoord = device_wrap(
                    Variable(coord[splitlist[i]:splitlist[i + 1]],
                             volatile=True).to(dtype=torch.float32,
                                               device='cpu'))

                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input, inputcoord)
                output = output.data.cpu().numpy()
                outputlist.append(output)
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[
                      :, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, shortname + '_feature.npy'),
                    feature_selected)
        # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        print([i_name, shortname])
        e = time.time()

        np.save(os.path.join(save_dir, shortname + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, shortname + '_lbb.npy'), lbb)
    end_time = time.time()

    print(('elapsed time is %3.2f seconds' % (end_time - start_time)))
    print()
    print()
