import pytest
import main
from pathlib2 import Path

try:
    import lzma
except ImportError:
    from backports import lzma
import tarfile
from test_xmlreport import compare_finding, compare_reports
import xmlreport
import numpy as np
import torch
import xml.etree.ElementTree as et


def ensure_testdata_unpacked(dataset="inputs"):
    res_dir = (Path(__file__).parent / "resources").absolute()
    inputs_dir = res_dir / dataset
    res_file = res_dir / "{}.tar.xz".format(dataset)
    if not inputs_dir.exists():
        with lzma.open(str(res_file), "r") as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(str(res_dir))
    assert inputs_dir.exists()
    return inputs_dir


def get_config(tmp_path, test_data_dir):
    cfg = {
        "detector_model": "net_detector",
        "detector_param": "./model/detector.ckpt",
        "classifier_model": "net_classifier",
        "classifier_param": "./model/classifier.ckpt",
        "n_gpu": 0,
        "n_worker_preprocessing": 6,
    }
    output_bbox_dir = tmp_path / "bbox"
    output_prep_dir = tmp_path / "prep"
    cfg.update(
        dict(
            datapath=str(test_data_dir),
            outputdir=str(tmp_path),
            output_bbox_dir=str(output_bbox_dir),
            output_prep_dir=str(output_prep_dir),
            use_existing_preprocessing=True,
        )
    )
    return cfg


def test_main(tmp_path,):
    test_data_dir = ensure_testdata_unpacked()
    cfg = get_config(tmp_path, test_data_dir)
    output_bbox_dir = tmp_path / "bbox"
    output_prep_dir = tmp_path / "prep"
    outputfile = tmp_path / "outfile.csv"
    crop_rects_outputfile = tmp_path / "croprects.csv"
    output_convert_debug_file = tmp_path / "convert_debug.json"

    main.main(
        outputfile=str(outputfile),
        crop_rects_outputfile=str(crop_rects_outputfile),
        output_convert_debug_file=str(output_convert_debug_file),
        skip_preprocessing=False,
        skip_detect=False,
        **cfg
    )

    assert output_bbox_dir.exists()
    assert output_prep_dir.exists()
    assert outputfile.exists()
    assert crop_rects_outputfile.exists()
    assert output_convert_debug_file.exists()

    # test if skip_detect works
    main.main(
        outputfile=str(outputfile),
        crop_rects_outputfile=str(crop_rects_outputfile),
        output_convert_debug_file=str(output_convert_debug_file),
        skip_preprocessing=False,
        skip_detect=True,
        **cfg
    )


def verify_lidc_inputs(results):
    assert results[2].imageinfo.seriesuid == "lidc-dcm"
    results[2].imageinfo.seriesuid = "lidc"
    assert len(results) == 3
    assert results[0] == results[1]
    assert results[0] == results[2]


def test_correct_top5(tmp_path,):
    test_data_dir = ensure_testdata_unpacked()
    cfg = get_config(tmp_path, test_data_dir)

    results_original = main.main(
        skip_detect=False,
        skip_preprocessing=False,
        classifier_max_nodules_to_include=5,
        classifier_num_nodules_for_cancer_decision=5,
        **cfg
    )
    verify_lidc_inputs(results_original)

    results_all = main.main(
        skip_detect=True,
        skip_preprocessing=True,
        classifier_max_nodules_to_include=None,
        classifier_num_nodules_for_cancer_decision=5,
        **cfg
    )
    verify_lidc_inputs(results_all)

    results_7 = main.main(
        skip_detect=True,
        skip_preprocessing=True,
        classifier_max_nodules_to_include=None,
        classifier_num_nodules_for_cancer_decision=7,
        **cfg
    )
    verify_lidc_inputs(results_7)

    # assert for all findings in results_original that they exist in order in results_all
    assert len(results_original[0].findings) < len(results_all[0].findings)
    for idx, finding in enumerate(results_original[0].findings):
        finding2 = results_all[0].findings[idx]
        compare_finding(finding, finding2)

    assert np.isclose(
        results_all[0].cancerinfo.casecancerprobability,
        results_original[0].cancerinfo.casecancerprobability,
        atol=1e-5,
    )
    assert not np.isclose(
        results_7[0].cancerinfo.casecancerprobability,
        results_original[0].cancerinfo.casecancerprobability,
        atol=1e-5,
    )

    assert (
        results_all[0].cancerinfo.referencenoduleids
        == results_original[0].cancerinfo.referencenoduleids
    )
    assert (
        results_7[0].cancerinfo.referencenoduleids
        != results_original[0].cancerinfo.referencenoduleids
    )
    assert len(results_7[0].cancerinfo.referencenoduleids) == 7
    assert len(results_all[0].cancerinfo.referencenoduleids) == 5
    assert len(results_original[0].cancerinfo.referencenoduleids) == 5


@pytest.mark.parametrize(
    ["nodules", "classifier_batch_size"],
    [
        (0, 20),
        (3, 20),
        (3, 2),
        (43, 20),
        pytest.param(
            43, 50,
            marks=[
                pytest.mark.xfail(reason="cuda out of memory"),
                pytest.mark.skipif(
                    not torch.cuda.is_available
                    or torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).total_memory
                    / 1024.0 ** 3
                    >= 6.0,
                    reason="cuda not available or GPU has more than 6 GB of VRAM required for test",
                ),
            ],
        )
    ],
)
def test_num_nodules(tmp_path, nodules, classifier_batch_size):
    test_data_dir = ensure_testdata_unpacked(dataset="inputs2")
    cfg = get_config(tmp_path, test_data_dir)
    results = main.main(
        skip_detect=False,
        skip_preprocessing=False,
        data_filter=r"nodules_{}.mhd".format(nodules),
        classifier_batch_size=classifier_batch_size,
        **cfg
    )
    assert len(results) == 1
    assert len(results[0].findings) == nodules
    assert len(results[0].cancerinfo.referencenoduleids) == min(nodules, 5)
    if nodules == 0:
        assert np.isclose(results[0].cancerinfo.casecancerprobability, 0)


def test_matches_ref_report(tmp_path):
    ref_file = Path(__file__).parent / "resources" / "ref.xml"
    assert ref_file.exists()
    ref_report = xmlreport.LungCadReport.from_xml(et.parse(str(ref_file)))
    test_data_dir = ensure_testdata_unpacked()
    cfg = get_config(tmp_path, test_data_dir)
    results = main.main(
        skip_detect=False,
        skip_preprocessing=False,
        data_filter=r".*.mhd",
        **cfg
    )
    assert len(results) == 1
    result = results[0]
    assert len(ref_report.findings) == 5
    assert len(result.findings) == 9
    ref_report.lungcad = result.lungcad
    result.findings = result.findings[:5]
    compare_reports(ref_report, result)
