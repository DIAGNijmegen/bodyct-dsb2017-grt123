import os

config = {
    'datapath': os.environ.get("INPUT_DIR", '/input/'),
    'outputdir': os.environ.get("OUTPUT_DIR", '/output/'),
    'output_bbox_dir': os.path.join(os.environ.get("OUTPUT_DIR", '/output/'), "bbox"),
    'output_prep_dir': os.path.join(os.environ.get("OUTPUT_DIR", '/output/'), "prep"),
    'outputfile': os.path.join(
        os.environ.get("OUTPUT_DIR", "/output/"),
        "prediction.csv"),
    'crop_rects_outputfile': os.path.join(
        os.environ.get("OUTPUT_DIR", "/output/"),
        "crop_rects.json"),

    'detector_model': 'net_detector',
    'detector_param': './model/detector.ckpt',
    'classifier_model': 'net_classifier',
    'classifier_param': './model/classifier.ckpt',
    'n_gpu': int(os.environ.get("N_GPUS", "1")),
    'n_worker_preprocessing': int(os.environ.get("N_PREPROCESSING_TASKS", "6")),
    'use_exsiting_preprocessing': True,
    'skip_preprocessing': False,
    'skip_detect': False
}
