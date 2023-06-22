from preprocessing import full_prep
from config_submit import config as config_submit

import numpy as np
import json
import os
import re
from datetime import datetime
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from data_detector import DataBowl3Detector, collate
from data_classifier import DataBowl3Classifier

from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
from convert_voxel_to_world import ConvertVoxelToWorld

import xmlreport


def main(datapath, outputdir, output_bbox_dir, output_prep_dir,
         detector_model="net_detector", detector_param="./model/detector.ckpt",
         classifier_model="net_classifier", classifier_param="./model/classifier.ckpt",
         n_gpu=1, n_worker_preprocessing=6, outputfile=None,
         crop_rects_outputfile=None, output_convert_debug_file=None,
         use_existing_preprocessing=True, skip_preprocessing=False, skip_detect=False,
         classifier_max_nodules_to_include=None, classifier_num_nodules_for_cancer_decision=5,
         classifier_batch_size=20,
         data_filter=None):
    execution_starttime = datetime.now()
    use_gpu = n_gpu > 0

    # Define the set of scans that we want to process
    testsplit = [f for f in os.listdir(datapath)
                 if os.path.isdir(os.path.join(datapath, f)) or os.path.splitext(f)[
                     1].lower() in (".mhd", ".mha")]
    if data_filter is not None:
        testsplit_original_size = len(testsplit)
        testsplit = [e for e in testsplit if re.match(data_filter, e)]
        print(("Matched {}/{} files in datapath using: {}\n{}".format(len(testsplit), testsplit_original_size, data_filter, testsplit)))

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
        print("Detecting...")
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

    # Free up nodule detection module resources
    del nod_net.module

    print("Applying case model...")

    casemodel = import_module(classifier_model.split('.py')[0])
    casenet = casemodel.CaseNet(topk=classifier_num_nodules_for_cancer_decision)
    config2 = casemodel.config
    checkpoint = torch.load(classifier_param, encoding="latin1")  # Uses old Python 2 pickle format using the latin1 encoding...
    casenet.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        torch.cuda.set_device(0)
        casenet = casenet.cuda()
    casenet = DataParallel(casenet)

    def test_casenet(model, testset):
        def custom_collate_fn(batch):
            if len(batch[0][0]) == 0:
                return None, None
            else:
                return default_collate(batch)

        data_loader = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True)
        model.eval()
        nodule_cancer_probabilities = {}
        predlist = []

        with torch.no_grad():
            for i, (x, coord) in enumerate(data_loader):
                # if nodules for case i are > 0
                if x is not None:
                    out_combined = None
                    if use_gpu:
                        coord = coord.cuda()
                        x = x.cuda()
                    num_nodules = x.size()[1]
                    for j in range(0, num_nodules, classifier_batch_size):
                        _, _, out = model(x[:, j:j+classifier_batch_size, :], coord[:, j:j+classifier_batch_size, :])
                        out_combined = torch.cat((out_combined, out.detach()), dim=1) if out_combined is not None else out.detach()

                    # compute image level cancer score
                    base_prob = torch.sigmoid(model.module.baseline)
                    casePred = 1 - torch.prod(1 - out_combined[:, :model.module.topk], dim=1) * (
                            1 - base_prob.expand(out_combined.size()[0]))

                    nodule_cancer_probabilities[i] = out_combined.data[0, :].cpu().numpy()
                    predlist.append(casePred.data.cpu().numpy())
                else:
                    nodule_cancer_probabilities[i] = []
                    predlist.append(torch.sigmoid(model.module.baseline).cpu().numpy())
        predlist = np.concatenate(predlist)
        return predlist, nodule_cancer_probabilities

    config2['bboxpath'] = output_bbox_dir
    config2['datadir'] = output_prep_dir
    config2['topk'] = classifier_max_nodules_to_include

    dataset = DataBowl3Classifier(testsplit, config2, phase='test')
    predlist, nodule_cancer_probabilities = test_casenet(casenet, dataset)
    predlist = predlist.T
    if predlist.ndim == 1:
        predlist = [predlist]

    nodule_cancer_probabilities = {testsplit[k]:v for k, v in list(nodule_cancer_probabilities.items())}

    cancer_probabilities = {k: v for k, v in zip(testsplit, predlist[0].tolist())}

    if outputfile is not None:
        anstable = np.concatenate([[testsplit], predlist], 0).T
        df = pandas.DataFrame(anstable)
        df.columns = ['id', 'cancer']
        df.to_csv(outputfile, index=False)

    if crop_rects_outputfile is not None:
        with open(crop_rects_outputfile, 'w') as f:
            json.dump(dataset.crop_rect_map, f, indent=4)

    converter = ConvertVoxelToWorld(preprocessing_info_dir=output_prep_dir,
                                    cropped_rects=dataset.crop_rect_map,
                                    output_file=output_convert_debug_file)

    # extract image infos from ConvertVoxelToWorld
    def extract_info(d):
        def reorder(e, typefn=float):
            return (typefn(e['x']), typefn(e['y']), typefn(e['z']))

        dimensions = reorder(d["original_shape"], typefn=int)
        origin = reorder(d["original_origin"])
        orientation = np.array([reorder(d["rotation_matrix_{}".format(e)]) for e in ["x", "y", "z"]]).flatten().tolist()
        voxelsize = reorder(d["original_spacing"])
        return [dimensions, orientation, origin, voxelsize]

    cparams = converter._conversion_parameters
    image_infos = {key: extract_info(cparams[key]) for key in list(cparams.keys())}

    # extract nodule confidences and compute nodule probabilities for all cases
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def filter_and_sort_confidences(conf_list, topk=None):
        topk = topk if topk is not None else len(conf_list)
        chosenid = conf_list.argsort()[::-1][:topk]
        return conf_list[chosenid]

    nodule_probs = {fname : sigmoid(filter_and_sort_confidences(dataset.candidate_box[dataset.src_file_names.index(fname)][:, 0], topk=dataset.topk)).tolist() for fname in testsplit}

    # construct output reports based on image-level findings
    time_diff = datetime.now() - execution_starttime
    computation_time = (time_diff.seconds + time_diff.microseconds / float(10 ** 6)) / len(image_infos)

    reports = []
    git_hash = xmlreport.get_current_git_hash()
    for seriesuid, (dimensions, orientation, origin, voxelsize) in list(image_infos.items()):
        lungcad = xmlreport.LungCad(revision=git_hash, name="grt123",
                          datetimeofexecution=execution_starttime.strftime("%m/%d/%Y %H:%M:%S"),
                          trainingset1="", trainingset2="", coordinatesystem="World",
                          computationtimeinseconds=computation_time)

        seriesuidplain = seriesuid.replace(".mhd", "").replace(".mha", "")
        imageinfo = xmlreport.ImageInfo(dimensions=dimensions, voxelsize=voxelsize, origin=origin,
                              orientation=orientation,
                              patientuid="", studyuid="", seriesuid=seriesuidplain)

        referencenoduleids = list(range(min(classifier_num_nodules_for_cancer_decision, len(nodule_probs[seriesuid]))))
        cancerinfo = xmlreport.CancerInfo(casecancerprobability=cancer_probabilities[seriesuid],
                                          referencenoduleids=referencenoduleids)


        findings = []
        for idx, cancer_prob in enumerate(nodule_cancer_probabilities[seriesuid]):
            coords = np.array([converter._coordinates[seriesuid][idx]["world_{}".format(e)] for e in ["x", "y", "z"]]).T
            center, extent = coords[1, :], coords[2, :] - coords[0, :]
            finding = xmlreport.Finding(id=idx, x=center[0], y=center[1], z=center[2],
                                        probability=nodule_probs[seriesuid][idx], diameter_mm=-1, volume_mm3=-1,
                                        extent=extent.tolist(), cancerprobability=float(cancer_prob))
            findings.append(finding)

        report = xmlreport.LungCadReport(lungcad, imageinfo, findings, cancerinfo=cancerinfo)

        # output xml reports
        report.to_file(fname=os.path.join(outputdir, seriesuid + ".xml"))

        reports.append(report)

    return reports


if __name__ == "__main__":
    print(config_submit)
    main(**config_submit)
