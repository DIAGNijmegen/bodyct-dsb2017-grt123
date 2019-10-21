import os

config = {
    'datapath': os.environ.get("INPUT_DIR", '/input/'),
    'outputdir': os.environ.get("OUTPUT_DIR", '/output/'),
    'output_bbox_dir': os.path.join(os.environ.get("OUTPUT_DIR", '/output/'), "bbox"),
    'output_prep_dir': os.path.join(os.environ.get("OUTPUT_DIR", '/output/'), "prep"),
    'outputfile': None,
    'crop_rects_outputfile': None,
    'output_convert_debug_file': None,
    'detector_model': 'net_detector',
    'detector_param': './model/detector.ckpt',
    'classifier_model': 'net_classifier',
    'classifier_param': './model/classifier.ckpt',
    'classifier_max_nodules_to_include': None,
    'classifier_num_nodules_for_cancer_decision': 5,
    'classifier_batch_size': 20,
    'n_gpu': int(os.environ.get("N_GPUS", "1")),
    'n_worker_preprocessing': int(os.environ.get("N_PREPROCESSING_TASKS", "6")),
    'use_existing_preprocessing': True,
    'skip_preprocessing': False,
    'skip_detect': False
}
