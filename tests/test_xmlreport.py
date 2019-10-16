import xmlreport
import xml.etree.ElementTree as ET
import numpy as np
import pytest


def compare_reports(reporta, reportb, origin_atol=1e-3, prob_atol=1e-4):
    assert reporta != reportb
    for attr in ["dimensions", "voxelsize", "orientation"]:
        a, b = getattr(reporta.imageinfo, attr), getattr(reportb.imageinfo, attr)
        assert isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            assert np.isclose(aa, bb, atol=1e-8)
    for attr in ["patientuid", "studyuid", "seriesuid"]:
        assert getattr(reporta.imageinfo, attr) == getattr(reportb.imageinfo, attr)

    assert np.allclose(
        reporta.imageinfo.origin, reportb.imageinfo.origin, atol=origin_atol
    )

    if reporta.cancerinfo is None:
        assert reportb.cancerinfo is None
    else:
        assert reporta.cancerinfo.referencenoduleids == reportb.cancerinfo.referencenoduleids
        assert np.isclose(reporta.cancerinfo.casecancerprobability, reportb.cancerinfo.casecancerprobability, atol=prob_atol)

    assert len(reporta.findings) == len(reportb.findings)
    for f1, f2 in zip(reporta.findings, reportb.findings):
        compare_finding(f1, f2, prob_atol=prob_atol)


def compare_finding(f1, f2, prob_atol=1e-5,):
    for attr in ["id", "x", "y", "z", "diameter_mm", "volume_mm3"]:
        assert getattr(f1, attr) == getattr(f2, attr)
    assert np.isclose(f1.probability, f2.probability, atol=prob_atol)
    a, b = f1.extent, f2.extent
    if a is None:
        assert b is None
    else:
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            assert np.isclose(aa, bb, atol=1e-8)
    if f1.cancerprobability is None:
        assert f2.cancerprobability is None
    else:
        assert np.isclose(
            f1.cancerprobability, f2.cancerprobability, atol=prob_atol
        )


def check_equivalence_and_properties(sampleA, sampleA2, sampleB):
    assert sampleA == sampleA
    assert sampleA == sampleA2
    assert sampleA != sampleB
    assert sampleA2 != sampleB
    for sample in (sampleA, sampleA2, sampleB):
        sample.validate()
        xmlreport.prettify(sample.xml_element())


def test_full_report_generation():
    finding = xmlreport.Finding(0, 2.0, 3.0, 4.0, 0.5, 20.0, 10.0)
    lungcad = xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    imageinfo = xmlreport.ImageInfo(
        dimensions=(0, 0, 0),
        voxelsize=(0.0, 0.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(0, 0, 0, 0, 0, 0, 0, 0, 0.5),
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )
    cancerinfo = xmlreport.CancerInfo(
        casecancerprobability=0.5, referencenoduleids=[1, 2, 3, 4, 5]
    )

    report = xmlreport.LungCadReport(
        lungcad, imageinfo, [finding, finding], cancerinfo=cancerinfo
    )
    report2 = xmlreport.LungCadReport(
        lungcad, imageinfo, [finding, finding], cancerinfo=cancerinfo
    )
    report3 = xmlreport.LungCadReport(lungcad, imageinfo, [finding])
    check_equivalence_and_properties(report, report2, report3)
    report4 = xmlreport.LungCadReport.from_xml(report.xml_element())
    check_equivalence_and_properties(report4, report2, report3)


def test_equivalence_finding():
    finding = xmlreport.Finding(
        0, 2.0, 3.0, 4.0, 0.5, 20.0, 10.0, extent=[1, 2, 3], cancerprobability=0.4
    )
    finding2 = xmlreport.Finding(
        0, 2.0, 3.0, 4.0, 0.5, 20.0, 10.0, extent=(1, 2, 3), cancerprobability=0.4
    )
    finding3 = xmlreport.Finding(
        0, 2.0, 3.0, 4.0, 1.5, 20.0, 10.0, extent=[1, 2, 3], cancerprobability=0.4
    )
    check_equivalence_and_properties(finding, finding2, finding3)
    finding4 = xmlreport.Finding.from_xml(finding.xml_element())
    check_equivalence_and_properties(finding4, finding2, finding3)


def test_equivalence_imageinfo():
    imageinfo = xmlreport.ImageInfo(
        dimensions=(0, 0, 0),
        voxelsize=(0.0, 0.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(0, 0, 0, 0, 0, 0, 0, 0, 0.5),
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )
    imageinfo2 = xmlreport.ImageInfo(
        dimensions=(0, 0, 0),
        voxelsize=(0.0, 0.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(0, 0, 0, 0, 0, 0, 0, 0, 0.5),
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )
    imageinfo3 = xmlreport.ImageInfo(
        dimensions=(0, 0, 0),
        voxelsize=(0.0, 1.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(0, 0, 0, 0, 0, 0, 0, 0, 0.5),
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )
    check_equivalence_and_properties(imageinfo, imageinfo2, imageinfo3)
    imageinfo4 = xmlreport.ImageInfo.from_xml(imageinfo.xml_element())
    check_equivalence_and_properties(imageinfo4, imageinfo2, imageinfo3)


def test_equivalence_lungcad():
    lungcad = xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    lungcad2 = xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    lungcad3 = xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD2",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    check_equivalence_and_properties(lungcad, lungcad2, lungcad3)
    lungcad4 = xmlreport.LungCad.from_xml(lungcad.xml_element())
    check_equivalence_and_properties(lungcad4, lungcad2, lungcad3)


@pytest.mark.parametrize("casecancerprobability", [0, 0.1, 0.51, 1])
@pytest.mark.parametrize(
    "referencenoduleids", [tuple(), [], [1], range(6), range(1, 5)]
)
def test_equivalence_cancerinfo(casecancerprobability, referencenoduleids):
    cancerinfo = xmlreport.CancerInfo(
        casecancerprobability=0.5, referencenoduleids=[1, 2, 3, 4, 5]
    )
    cancerinfo2 = xmlreport.CancerInfo(
        casecancerprobability=0.5, referencenoduleids=(1, 2, 3, 4, 5)
    )
    cancerinfo3 = xmlreport.CancerInfo(
        casecancerprobability=casecancerprobability,
        referencenoduleids=referencenoduleids,
    )
    assert isinstance(cancerinfo2.referencenoduleids, tuple)
    check_equivalence_and_properties(cancerinfo, cancerinfo2, cancerinfo3)
    cancerinfo4 = xmlreport.CancerInfo.from_xml(cancerinfo.xml_element())
    check_equivalence_and_properties(cancerinfo4, cancerinfo2, cancerinfo3)


@pytest.mark.parametrize(
    "cancerinfo",
    [
        None,
        xmlreport.CancerInfo(
            casecancerprobability=0.5, referencenoduleids=[1, 2, 3, 4, 5]
        ),
    ],
)
def test_writing_and_reading_lungcadreports_xml(tmp_path, cancerinfo):
    testfile = str(tmp_path / "out.xml")

    findings = []
    for i in range(3):
        findings.append(xmlreport.Finding(i, 2.0, 3.0, 4.0, 0.5, 20.0, 10.0))
    findings.append(xmlreport.Finding(3, 1, 2, 3, 0.5, 0.1, -1, (2, 3, 4), 0.7))
    lungcad = xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    imageinfo = xmlreport.ImageInfo(
        dimensions=[0, 0, 0],
        voxelsize=[0.0, 0.0, 0.0],
        origin=[0.0, 0.0, 0.0],
        orientation=[0, 0, 0, 0, 0, 0, 0, 0, 0.5],
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )

    report = xmlreport.LungCadReport(
        lungcad, imageinfo, findings, cancerinfo=cancerinfo
    )

    with open(testfile, "w") as f:
        ET.ElementTree(report.xml_element()).write(
            f, encoding="UTF-8", xml_declaration=True
        )
    reportB = xmlreport.LungCadReport.from_xml(ET.parse(testfile))

    compare_reports(report, reportB)


def test_get_current_git_hash():
    hash = xmlreport.get_current_git_hash()
    lungcad = xmlreport.LungCad(
        revision=hash,
        name="GPUCAD",
        datetimeofexecution="01/01/1900 00:00:00",
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )
    lungcad.validate()
