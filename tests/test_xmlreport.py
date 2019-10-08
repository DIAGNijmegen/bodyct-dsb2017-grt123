import xmlreport
import xml.etree.ElementTree as ET
import numpy as np


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

    assert np.allclose(reporta.imageinfo.origin, reportb.imageinfo.origin, atol=origin_atol)

    assert len(reporta.findings) == len(reportb.findings)
    for f1, f2 in zip(reporta.findings, reportb.findings):
        for attr in ["id", "x", "y", "z", "diameter_mm", "volume_mm3"]:
            assert getattr(f1, attr) == getattr(f2, attr)
        assert np.isclose(f1.probability, f2.probability, atol=prob_atol)


def check_equivalence_and_properties(sampleA, sampleA2, sampleB):
    assert sampleA == sampleA
    assert sampleA == sampleA2
    assert sampleA != sampleB
    assert sampleA2 != sampleB
    for sample in (sampleA, sampleA2, sampleB):
        sample.validate()
        xmlreport.prettify(sample.xml_element())


def test_full_report_generation():
    finding = xmlreport.Finding(0, 2., 3., 4., 0.5, 20., 10.)
    lungcad = xmlreport.LungCad(revision= "", name="GPUCAD", datetimeofexecution="", trainingset1="",
                                trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    imageinfo = xmlreport.ImageInfo(dimensions=(0,0,0), voxelsize=(0.,0.,0.), origin=(0.,0.,0.),
                                    orientation=(0,0,0,0,0,0,0,0,0.5), patientuid="34.2.32",
                                    studyuid="232.32.3", seriesuid="424.35")

    report = xmlreport.LungCadReport(lungcad, imageinfo, [finding, finding])
    report2 = xmlreport.LungCadReport(lungcad, imageinfo, [finding, finding])
    report3 = xmlreport.LungCadReport(lungcad, imageinfo, [finding])
    check_equivalence_and_properties(report, report2, report3)
    report4 = xmlreport.LungCadReport.from_xml(report.xml_element())
    check_equivalence_and_properties(report4, report2, report3)


def test_equivalence_finding():
    finding = xmlreport.Finding(0, 2., 3., 4., 0.5, 20., 10.)
    finding2 = xmlreport.Finding(0, 2., 3., 4., 0.5, 20., 10.)
    finding3 = xmlreport.Finding(0, 2., 3., 4., 1.5, 20., 10.)
    check_equivalence_and_properties(finding, finding2, finding3)
    finding4 = xmlreport.Finding.from_xml(finding.xml_element())
    check_equivalence_and_properties(finding4, finding2, finding3)


def test_equivalence_imageinfo():
    imageinfo = xmlreport.ImageInfo(dimensions=(0,0,0), voxelsize=(0.,0.,0.), origin=(0.,0.,0.),
                                    orientation=(0,0,0,0,0,0,0,0,0.5), patientuid="34.2.32",
                                    studyuid="232.32.3", seriesuid="424.35")
    imageinfo2 = xmlreport.ImageInfo(dimensions=(0,0,0), voxelsize=(0.,0.,0.), origin=(0.,0.,0.),
                                    orientation=(0,0,0,0,0,0,0,0,0.5), patientuid="34.2.32",
                                    studyuid="232.32.3", seriesuid="424.35")
    imageinfo3 = xmlreport.ImageInfo(dimensions=(0,0,0), voxelsize=(0.,1.,0.), origin=(0.,0.,0.),
                                    orientation=(0,0,0,0,0,0,0,0,0.5), patientuid="34.2.32",
                                    studyuid="232.32.3", seriesuid="424.35")
    check_equivalence_and_properties(imageinfo, imageinfo2, imageinfo3)
    imageinfo4 = xmlreport.ImageInfo.from_xml(imageinfo.xml_element())
    check_equivalence_and_properties(imageinfo4, imageinfo2, imageinfo3)


def test_equivalence_lungcad():
    lungcad = xmlreport.LungCad(revision= "", name="GPUCAD", datetimeofexecution="", trainingset1="",
                                trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    lungcad2 = xmlreport.LungCad(revision= "", name="GPUCAD", datetimeofexecution="", trainingset1="",
                                 trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    lungcad3 = xmlreport.LungCad(revision= "", name="GPUCAD2", datetimeofexecution="", trainingset1="",
                                 trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    check_equivalence_and_properties(lungcad, lungcad2, lungcad3)
    lungcad4 = xmlreport.LungCad.from_xml(lungcad.xml_element())
    check_equivalence_and_properties(lungcad4, lungcad2, lungcad3)


def test_writing_and_reading_lungcadreports_xml(tmp_path):
    testfile = str(tmp_path / "out.xml")

    findings = []
    for i in range(3):
        findings.append(xmlreport.Finding(i, 2., 3., 4., 0.5, 20., 10.))
    lungcad = xmlreport.LungCad(revision= "", name="GPUCAD", datetimeofexecution="", trainingset1="",
                                trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    imageinfo = xmlreport.ImageInfo(dimensions=[0,0,0], voxelsize=[0.,0.,0.], origin=[0.,0.,0.],
                                    orientation=[0,0,0,0,0,0,0,0,0.5], patientuid="34.2.32",
                                    studyuid="232.32.3", seriesuid="424.35")

    report = xmlreport.LungCadReport(lungcad, imageinfo, findings)

    with open(testfile, "w") as f:
        ET.ElementTree(report.xml_element()).write(f, encoding="UTF-8", xml_declaration=True)
    reportB = xmlreport.LungCadReport.from_xml(ET.parse(testfile))

    compare_reports(report, reportB)

