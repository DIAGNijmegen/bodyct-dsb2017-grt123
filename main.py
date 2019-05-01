from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector, collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
from convert_voxel_to_world import ConvertVoxelToWorld

use_gpu = config_submit['n_gpu'] > 0

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    testsplit = full_prep(datapath, prep_result_path,
                          n_worker=config_submit['n_worker_preprocessing'],
                          use_existing=config_submit[
                              'use_exsiting_preprocessing'])
else:
    testsplit = os.listdir(datapath)
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

if use_gpu:
    torch.cuda.set_device(0)
    nod_net = nod_net.cuda()
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
# testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    print "Detecting..."
    margin = 32
    sidelen = 144

    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen, config1['max_stride'], config1['stride'],
                             margin, pad_value=config1['pad_value'])

    dataset = DataBowl3Detector(testsplit, config1, phase='test',
                                split_comber=split_comber)
    test_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=0, pin_memory=False,
                             collate_fn=collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path, config1,
                n_gpu=config_submit['n_gpu'])

print "Applying case model..."

casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])
casenet.load_state_dict(checkpoint['state_dict'])

if use_gpu:
    torch.cuda.set_device(0)
    casenet = casenet.cuda()
casenet = DataParallel(casenet)

filename = config_submit['outputfile']


def test_casenet(model, testset):
    data_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    model.eval()
    predlist = []

    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    with torch.no_grad():
        for i, (x, coord) in enumerate(data_loader):
            coord = Variable(coord, volatile=True)
            x = Variable(x, volatile=True)

            if use_gpu:
                coord = coord.cuda()
                x = x.cuda()

            nodulePred, casePred, _ = model(x, coord)
            predlist.append(casePred.data.cpu().numpy())
            # print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist


config2['bboxpath'] = bbox_result_path
config2['datadir'] = prep_result_path

dataset = DataBowl3Classifier(testsplit, config2, phase='test')
predlist = test_casenet(casenet, dataset).T
if predlist.ndim == 1:
    predlist = [predlist]
anstable = np.concatenate([[testsplit], predlist], 0).T
df = pandas.DataFrame(anstable)
df.columns = ['id', 'cancer']
df.to_csv(filename, index=False)

import json

with open(config_submit['crop_rects_outputfile'], 'wb') as f:
    json.dump(dataset.crop_rect_map, f, indent=4)
testsplit = ''.join(testsplit)
preprocessing_info_dir = os.path.join(
    os.environ.get("OUTPUT_DIR", "/output/"))
ConvertVoxelToWorld(preprocessing_info_dir=preprocessing_info_dir,
                    cropped_rects=dataset.crop_rect_map,
                    output_path=os.environ.get("OUTPUT_DIR"))
