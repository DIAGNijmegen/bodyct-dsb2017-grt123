from typing import List, Tuple, Union, Optional
from datetime import datetime
from pathlib import Path

import copy
import pytest

import xmlreport


@pytest.fixture
def basic_cancerinfo() -> xmlreport.CancerInfo:
    return xmlreport.CancerInfo(
        casecancerprobability=0.5,
        referencenoduleids=[
            0,
        ],
    )


@pytest.fixture
def basic_finding() -> xmlreport.Finding:
    return xmlreport.Finding(
        id=0,
        x=2.0,
        y=3.0,
        z=4.0,
        probability=0.5,
        diameter_mm=20.0,
        volume_mm3=10.0,
        extent=[1, 2, 3],
    )


@pytest.fixture
def basic_imageinfo() -> xmlreport.ImageInfo:
    return xmlreport.ImageInfo(
        dimensions=(0, 0, 0),
        voxelsize=(0.0, 0.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        orientation=(0, 0, 0, 0, 0, 0, 0, 0, 0.5),
        patientuid="34.2.32",
        studyuid="232.32.3",
        seriesuid="424.35",
    )


@pytest.fixture
def basic_lungcad() -> xmlreport.LungCad:
    return xmlreport.LungCad(
        revision="abcdef01234567890aaaaaaaaaaaaaaaaaaaaaaa",
        name="GPUCAD",
        datetimeofexecution=datetime.now(),
        trainingset1="",
        trainingset2="",
        coordinatesystem="World",
        computationtimeinseconds=33.0,
    )


def check_equivalence_and_properties(
    samplea: xmlreport.XMLGeneratable,
    samplea2: xmlreport.XMLGeneratable,
    sampleb: xmlreport.XMLGeneratable,
):
    assert samplea == samplea
    assert samplea.is_similar(samplea)
    assert samplea == samplea2
    assert samplea.is_similar(samplea2)
    assert samplea != sampleb
    assert not samplea.is_similar(sampleb)
    assert samplea2 != sampleb
    assert not samplea2.is_similar(sampleb)
    for sample in (samplea, samplea2, sampleb):
        sample.validate()
        sample.to_xml()
        sample.to_json()
        sample.__str__()


def test_full_report_generation(
    basic_finding: xmlreport.Finding,
    basic_lungcad: xmlreport.LungCad,
    basic_imageinfo: xmlreport.ImageInfo,
    basic_cancerinfo: xmlreport.CancerInfo,
):
    report = xmlreport.LungCadReport(
        basic_lungcad,
        basic_imageinfo,
        [basic_finding, basic_finding],
        cancerinfo=basic_cancerinfo,
    )
    report2 = xmlreport.LungCadReport(
        basic_lungcad,
        basic_imageinfo,
        [basic_finding, basic_finding],
        cancerinfo=basic_cancerinfo,
    )
    report3 = xmlreport.LungCadReport(
        basic_lungcad, basic_imageinfo, [basic_finding]
    )
    check_equivalence_and_properties(report, report2, report3)
    report4 = xmlreport.LungCadReport.from_xml(report.to_xml())
    check_equivalence_and_properties(report4, report2, report3)


@pytest.mark.parametrize(
    "cancerprobability, noduletype",
    (
        (0.4, None),
        (None, "Solid"),
    ),
)
def test_equivalence_finding(
    basic_finding: xmlreport.Finding,
    cancerprobability: Optional[float],
    noduletype: Optional[str],
):
    finding = copy.deepcopy(basic_finding)
    finding.cancerprobability = cancerprobability
    finding.noduletype = noduletype

    finding2 = copy.deepcopy(finding)
    finding2.extent = (1, 2, 3)

    finding3 = copy.deepcopy(finding2)
    finding3.probability = 1.5

    finding4 = xmlreport.Finding.from_xml(finding2.to_xml())

    finding5 = copy.deepcopy(finding)
    finding5.cancerprobability = 0.999
    finding5.noduletype = "UNKNOWN"

    finding6 = copy.deepcopy(finding5)
    finding6.nodulesegmentationseedvectorbegin = (0, 0, 0)
    finding6.nodulesegmentationseedvectorend = [2, 3, 4]

    finding7 = copy.deepcopy(finding5)
    finding7.nodulesegmentationseedvectorbegin = [0, 0, 0]
    finding7.nodulesegmentationseedvectorend = (2, 3, 4)

    check_equivalence_and_properties(finding, finding2, finding3)
    check_equivalence_and_properties(finding4, finding2, finding3)
    assert finding != finding5
    assert finding != finding6
    check_equivalence_and_properties(finding6, finding7, finding)

    finding8 = copy.deepcopy(finding7)
    finding8.mass_mg = 2.3
    finding8.majoraxis_mm = 12.0
    finding8.minoraxis_mm = 11.0

    f = copy.deepcopy(finding8)
    f.mass_mg = 0
    assert finding8 != f
    f = copy.deepcopy(finding8)
    f.majoraxis_mm = 0
    assert finding8 != f
    f = copy.deepcopy(finding8)
    f.minoraxis_mm = 0
    assert finding8 != f

    finding9 = copy.deepcopy(finding8)
    finding9.solidcoresegmentationseedvectorbegin = (1, 2, 3)
    finding9.solidcoresegmentationseedvectorend = [4, 5, 6]
    finding9.noduletype = "PartSolid"
    finding9.meandensity_hu = 200
    finding9.corediameter_mm = 1.0
    finding9.coremass_mg = 2.0
    finding9.coremajoraxis_mm = 3.0
    finding9.coreminoraxis_mm = 4.0
    finding9.corevolume_mm3 = 5.0
    finding9.coremeandensity_hu = 120
    finding9.averagediameter_mm = 8.0
    finding9.coreaveragediameter_mm = 9.0

    finding10 = copy.deepcopy(finding9)
    finding10.solidcoresegmentationseedvectorbegin = [1, 2, 3]
    finding10.solidcoresegmentationseedvectorend = (4, 5, 6)

    check_equivalence_and_properties(finding9, finding10, finding8)

    for attr in [
        "meandensity_hu",
        "corediameter_mm",
        "coremass_mg",
        "coremajoraxis_mm",
        "coreminoraxis_mm",
        "corevolume_mm3",
        "coremeandensity_hu",
        "averagediameter_mm",
        "coreaveragediameter_mm",
    ]:
        f = copy.deepcopy(finding9)
        setattr(f, attr, 0)
        assert f != finding9
        assert not f.is_similar(finding9)


@pytest.mark.parametrize(
    "vbegin, vend",
    (((0, 0, 0), None), (None, (1, 3, 4)), ((1, 2, 3, 4), (0, 0, 0))),
)
def test_bad_nodulesegmentationseedvector_findings(
    basic_finding: xmlreport.Finding,
    vbegin: Optional[Tuple],
    vend: Optional[Tuple],
):
    with pytest.raises(ValueError):
        basic_finding.nodulesegmentationseedvectorbegin = vbegin
        basic_finding.nodulesegmentationseedvectorend = vend
        basic_finding.validate()


@pytest.mark.parametrize(
    "vbegin, vend",
    (
        ((0, 0, 0), None),
        (None, (1, 3, 4)),
        ((1, 2, 3, 4), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    ),
)
def test_bad_solidcoresegmentationseedvector_findings(
    basic_finding: xmlreport.Finding,
    vbegin: Optional[Tuple],
    vend: Optional[Tuple],
):
    with pytest.raises(ValueError):
        basic_finding.solidcoresegmentationseedvectorbegin = vbegin
        basic_finding.solidcoresegmentationseedvectorend = vend
        basic_finding.validate()


@pytest.mark.parametrize(
    "cancer_info",
    (
        xmlreport.CancerInfo(
            casecancerprobability=0.2,
            referencenoduleids=(
                0,
                1,
                2,
            ),
        ),
        xmlreport.CancerInfo(casecancerprobability=0.2, referencenoduleids=()),
        xmlreport.CancerInfo(
            casecancerprobability=0.2, referencenoduleids=(0,)
        ),
    ),
)
def test_cancer_info(tmp_path: Path, cancer_info: xmlreport.CancerInfo):
    tmp_file = tmp_path / "test.xml"
    cancer_info.to_file(tmp_file)
    ref_cinfo = xmlreport.CancerInfo.from_file(tmp_file)
    assert cancer_info.is_similar(ref_cinfo)


def test_cancer_info_invalid_refids(
    basic_finding: xmlreport.Finding,
    basic_imageinfo: xmlreport.ImageInfo,
    basic_lungcad: xmlreport.LungCad,
):
    cancerinfo = xmlreport.CancerInfo(
        casecancerprobability=0.2,
        referencenoduleids=(99,),
    )
    with pytest.raises(ValueError):
        xmlreport.LungCadReport(
            lungcad=basic_lungcad,
            imageinfo=basic_imageinfo,
            findings=[basic_finding],
            cancerinfo=cancerinfo,
        )


def test_equivalence_imageinfo(basic_imageinfo: xmlreport.ImageInfo):
    imageinfo = basic_imageinfo
    imageinfo2 = basic_imageinfo
    imageinfo3 = copy.deepcopy(basic_imageinfo)
    imageinfo3.voxelsize = (0.0, 1.0, 0.0)
    check_equivalence_and_properties(imageinfo, imageinfo2, imageinfo3)
    imageinfo4 = xmlreport.ImageInfo.from_xml(imageinfo.to_xml())
    check_equivalence_and_properties(imageinfo4, imageinfo2, imageinfo3)


def test_equivalence_lungcad(basic_lungcad: xmlreport.LungCad):
    lungcad = basic_lungcad
    lungcad2 = basic_lungcad
    lungcad3 = copy.deepcopy(basic_lungcad)
    lungcad3.name = "GPUCAD2"
    check_equivalence_and_properties(lungcad, lungcad2, lungcad3)
    lungcad4 = xmlreport.LungCad.from_xml(lungcad.to_xml())
    check_equivalence_and_properties(lungcad4, lungcad2, lungcad3)


@pytest.mark.parametrize("casecancerprobability", [0, 0.1, 0.51, 1])
@pytest.mark.parametrize(
    "referencenoduleids", [tuple(), [], [1], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4]]
)
def test_equivalence_cancerinfo(
    casecancerprobability: Union[int, float],
    referencenoduleids: List[Union[Tuple[int, ...], List[int]]],
):
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
    cancerinfo4 = xmlreport.CancerInfo.from_xml(cancerinfo.to_xml())
    check_equivalence_and_properties(cancerinfo4, cancerinfo2, cancerinfo3)


@pytest.mark.parametrize(
    "cancerinfo",
    [
        None,
        xmlreport.CancerInfo(
            casecancerprobability=0.5,
            referencenoduleids=(
                1,
                3,
            ),
        ),
    ],
)
def test_writing_and_reading_lungcadreports_xml(
    tmp_path: Path,
    basic_lungcad: xmlreport.LungCad,
    basic_imageinfo: xmlreport.ImageInfo,
    cancerinfo: xmlreport.CancerInfo,
):
    testfile = tmp_path / "out.xml"

    findings = []
    for i in range(3):
        findings.append(xmlreport.Finding(i, 2.0, 3.0, 4.0, 0.5, 20.0, 10.0))
    findings.append(
        xmlreport.Finding(3, 1, 2, 3, 0.5, 0.1, -1, (2, 3, 4), 0.7)
    )
    report = xmlreport.LungCadReport(
        lungcad=basic_lungcad,
        imageinfo=basic_imageinfo,
        findings=findings,
        cancerinfo=cancerinfo,
    )
    report.to_file(testfile)
    reportb = xmlreport.LungCadReport.from_file(testfile)

    assert report.is_similar(other=reportb)


def test_get_current_git_hash(basic_lungcad: xmlreport.LungCad):
    revision_hash = xmlreport.get_current_git_hash(
        Path(__file__).parent.parent / ".git"
    )
    lungcad = basic_lungcad
    lungcad.revision = revision_hash
    lungcad.validate()


@pytest.mark.parametrize("suffix", ("xml", "json"))
def test_writing_and_reading_xml_json_report_files(
    tmp_path: Path, suffix: str
):
    resources_dir = Path(__file__).parent / "resources"
    test_file = tmp_path / f"testout.{suffix}"
    count = 0
    for xml_file in resources_dir.glob("**/*.xml"):
        report = xmlreport.LungCadReport.from_file(fname=xml_file)
        report.to_file(test_file)
        reportb = xmlreport.LungCadReport.from_file(fname=test_file)
        assert report == reportb
        count += 1
    assert count > 0
