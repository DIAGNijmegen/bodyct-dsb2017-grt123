from preprocessing import full_prep
from config_submit import config as config_submit

import numpy as np
import json
import os
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_detector import DataBowl3Detector, collate
from data_classifier import DataBowl3Classifier

from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
from convert_voxel_to_world import ConvertVoxelToWorld


def main(datapath, outputdir, output_bbox_dir, output_prep_dir, outputfile,
         nodule_specific_prediction_file, crop_rects_outputfile,
         detector_model, detector_param, classifier_model, classifier_param,
         n_gpu, n_worker_preprocessing,
         use_existing_preprocessing=True, skip_preprocessing=False, skip_detect=False, ):
    use_gpu = n_gpu > 0

    # Define the set of scans that we want to process
    testsplit = [f for f in os.listdir(datapath)
                 if os.path.isdir(os.path.join(datapath, f)) or os.path.splitext(f)[
                     1].lower() in (".mhd", ".mha")]
    # If there are no mhd or mha files and no folders in the input dir, we assume that a folder with dcm files is provided
    if not testsplit:
        testsplit = [os.path.basename(datapath)]
        datapath = os.path.dirname(datapath)

    if not os.path.exists(output_prep_dir):
        os.mkdir(output_prep_dir)

    if not skip_preprocessing:
        full_prep(datapath, testsplit, output_prep_dir,
                  n_worker=n_worker_preprocessing,
                  use_existing=use_existing_preprocessing)

    nodmodel = import_module(detector_model.split('.py')[0])
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(detector_param)
    nod_net.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        torch.cuda.set_device(0)
        nod_net = nod_net.cuda()
    nod_net = DataParallel(nod_net)

    if not os.path.exists(output_bbox_dir):
        os.mkdir(output_bbox_dir)

    if not skip_detect:
        print "Detecting..."
        margin = 32
        sidelen = 144

        config1['datadir'] = output_prep_dir
        split_comber = SplitComb(sidelen, config1['max_stride'], config1['stride'],
                                 margin, pad_value=config1['pad_value'])

        dataset = DataBowl3Detector(testsplit, config1, phase='test',
                                    split_comber=split_comber)
        test_loader = DataLoader(dataset, batch_size=1,
                                 shuffle=False, num_workers=0, pin_memory=False,
                                 collate_fn=collate)

        test_detect(test_loader, nod_net, get_pbb, output_bbox_dir, config1,
                    n_gpu=n_gpu)

    print "Applying case model..."

    casemodel = import_module(classifier_model.split('.py')[0])
    casenet = casemodel.CaseNet(topk=5)
    config2 = casemodel.config
    checkpoint = torch.load(classifier_param)
    casenet.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        torch.cuda.set_device(0)
        casenet = casenet.cuda()
    casenet = DataParallel(casenet)


    def test_casenet(model, testset):
        data_loader = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
        model.eval()
        nodule_specific_probability_list = []
        predlist = []

        #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
        with torch.no_grad():
            for i, (x, coord) in enumerate(data_loader):
                coord = Variable(coord, volatile=True)
                x = Variable(x, volatile=True)

                if use_gpu:
                    coord = coord.cuda()
                    x = x.cuda()

                nodulePred, casePred, out = model(x, coord)
                nodule_specific_probability_list.append(out.data.cpu().numpy())
                predlist.append(casePred.data.cpu().numpy())
                # print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
        nodule_specific_probability_list = np.concatenate(nodule_specific_probability_list)
        predlist = np.concatenate(predlist)
        return predlist, nodule_specific_probability_list


    config2['bboxpath'] = output_bbox_dir
    config2['datadir'] = output_prep_dir

    dataset = DataBowl3Classifier(testsplit, config2, phase='test')
    predlist, nodule_specific_probability_list = test_casenet(casenet, dataset)
    predlist = predlist.T
    nodule_specific_probability_list = nodule_specific_probability_list.T
    if predlist.ndim == 1:
        predlist = [predlist]
    if nodule_specific_probability_list.ndim == 1:
        nodule_specific_probability_list = [nodule_specific_probability_list]

    anstable = np.concatenate([[testsplit], predlist], 0).T
    df = pandas.DataFrame(anstable)
    df.columns = ['id', 'cancer']
    df.to_csv(outputfile, index=False)

    # df = pandas.DataFrame(nodule_specific_probability_list)
    # df.columns = ['cancer_score']
    # df.to_csv(nodule_specific_prediction_file, index=False)

    with open(crop_rects_outputfile, 'wb') as f:
        json.dump(dataset.crop_rect_map, f, indent=4)
    ConvertVoxelToWorld(preprocessing_info_dir=output_prep_dir,
                        cropped_rects=dataset.crop_rect_map,
                        output_path=outputdir)


if __name__ == "__main__":
    print(config_submit)
    main(**config_submit)
