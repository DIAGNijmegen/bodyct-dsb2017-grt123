import main
from pathlib2 import Path
try:
    import lzma
except ImportError:
    from backports import lzma
import tarfile


def ensure_testdata_unpacked():
    res_dir = (Path(__file__).parent / "resources").absolute()
    inputs_dir = res_dir / "inputs"
    res_file = res_dir / "inputs.tar.xz"
    if not inputs_dir.exists():
        with lzma.open(str(res_file), "r") as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(str(res_dir))
    assert inputs_dir.exists()
    return inputs_dir


basic_config = {
    'detector_model': 'net_detector',
    'detector_param': './model/detector.ckpt',
    'classifier_model': 'net_classifier',
    'classifier_param': './model/classifier.ckpt',
    'n_gpu': 0,
    'n_worker_preprocessing': 6,
}


def test_main(tmp_path, ):
    test_data_dir = ensure_testdata_unpacked()
    output_bbox_dir = tmp_path / "bbox"
    output_prep_dir = tmp_path / "prep"
    outputfile = tmp_path / "outfile.csv"
    crop_rects_outputfile = tmp_path / "croprects.csv"
    output_convert_debug_file = tmp_path / "convert_debug.json"

    main.main(datapath=str(test_data_dir),
              outputdir=str(tmp_path),
              output_bbox_dir=str(output_bbox_dir),
              output_prep_dir=str(output_prep_dir),
              outputfile=str(outputfile),
              crop_rects_outputfile=str(crop_rects_outputfile),
              output_convert_debug_file=str(output_convert_debug_file),
              use_existing_preprocessing=True,
              skip_preprocessing=False,
              skip_detect=False,
              **basic_config
              )

    assert output_bbox_dir.exists()
    assert output_prep_dir.exists()
    assert outputfile.exists()
    assert crop_rects_outputfile.exists()
    assert output_convert_debug_file.exists()

    # test if skip_detect works
    main.main(datapath=str(test_data_dir),
              outputdir=str(tmp_path),
              output_bbox_dir=str(output_bbox_dir),
              output_prep_dir=str(output_prep_dir),
              outputfile=str(outputfile),
              crop_rects_outputfile=str(crop_rects_outputfile),
              output_convert_debug_file=str(output_convert_debug_file),
              use_existing_preprocessing=True,
              skip_preprocessing=False,
              skip_detect=True,
              **basic_config
              )
