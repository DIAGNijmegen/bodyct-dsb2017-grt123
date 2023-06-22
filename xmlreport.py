from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
from xml.etree import ElementTree
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import json
import numpy as np


def indent(elem: ElementTree.Element, level: int = 0, more_sibs: bool = False):
    i = "\n"
    if level:
        i += (level - 1) * "  "
    num_kids = len(elem)
    if num_kids:
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
            if level:
                elem.text += "  "
        count = 0
        for kid in elem:
            indent(kid, level + 1, count < num_kids - 1)
            count += 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            if more_sibs:
                elem.tail += "  "
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            if more_sibs:
                elem.tail += "  "


def get_current_git_hash(git_dir: Path = Path(__file__).parent / ".git") -> str:
    head_file = git_dir / "HEAD"
    with open(str(head_file), "r") as f:
        content = f.read().strip().split(" ")
    if "ref:" != content[0]:
        return content[0]
    with open(str(git_dir / Path(content[1])), "r") as f:
        git_hash = f.read().strip()
    return git_hash


def instanceofcheck(obj: object, typ: Any):
    if not isinstance(obj, typ):
        raise ValueError(f"Object type ({obj}) is not of type: {typ}")


def lencheck(obj: object, length: int):
    instanceofcheck(obj, (list, tuple))
    assert isinstance(obj, (list, tuple))
    if not len(obj) == length:
        raise ValueError(f"Object length is ({len(obj)}), expected: {length}")


def matchcheck(obj: str, pattern: Pattern):
    if not re.match(pattern, obj.strip()):
        raise ValueError(f"{obj} does not match regex. pattern: {pattern}")


def none_guard_element(
    obj: Union[ElementTree.Element, ElementTree.ElementTree], name: str
) -> ElementTree.Element:
    result = obj.find("./" + name)
    if result is None:
        raise ValueError(f"Missing required argument: {name}")
    return result


def get_xml_value(
    xml: Union[ElementTree.Element, ElementTree.ElementTree],
    name: str,
    required: bool = True,
) -> str:
    result = xml.find("./" + name)
    if result is None or result.text is None:
        if required:
            raise ValueError(f"Missing required argument: {name}")
        return ""
    if result.text.lower() == "unknown":
        return ""
    return result.text


def get_xml_value_or_none(
    xml: Union[ElementTree.Element, ElementTree.ElementTree], name: str
) -> Optional[str]:
    result = xml.find("./" + name)
    if result is None:
        return None
    value = get_xml_value(xml=xml, name=name, required=False)
    if value == "":
        return None
    return value


def optional_test(
    a: Optional[Any], b: Optional[Any], test_fn: Callable
) -> bool:
    if a is not None and b is not None:
        return test_fn(a, b)
    return a is b


class XMLGeneratable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractmethod
    def is_similar(self, other: "XMLGeneratable", atol: float = 1e-8) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def to_xml(self) -> ElementTree.Element:
        raise NotImplementedError()

    @abstractmethod
    def to_json(self) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "XMLGeneratable":
        raise NotImplementedError()

    @staticmethod
    def from_json(data: Dict) -> "XMLGeneratable":
        raise NotImplementedError()

    @classmethod
    def from_file(cls, fname: Path) -> "XMLGeneratable":
        fname = Path(fname)
        if fname.suffix == ".xml":
            return cls.from_xml(ElementTree.parse(str(fname)))
        elif fname.suffix == ".json":
            with open(fname, "r") as f:
                data = json.load(f)
            return cls.from_json(data)
        else:
            raise NotImplementedError(
                f"Unsupported file suffix: {fname.suffix}, supported are: .xml, .csv"
            )

    def to_file(self, fname: Path):
        fname = Path(fname)
        if fname.suffix == ".xml":
            with open(fname, "wb") as f:
                tree = ElementTree.ElementTree(self.to_xml())
                indent(tree.getroot())
                tree.write(f, encoding="UTF-8", xml_declaration=True)
        elif fname.suffix == ".json":
            with open(fname, "w") as fp:
                json.dump(obj=self.to_json(), fp=fp)
        else:
            raise NotImplementedError(
                f"Unsupported file suffix: {fname.suffix}, supported are: .xml, .csv"
            )


class LungCadReport(XMLGeneratable):
    def __init__(
        self,
        lungcad: "LungCad",
        imageinfo: "ImageInfo",
        findings: List["Finding"],
        cancerinfo: Optional["CancerInfo"] = None,
    ):
        self.lungcad = lungcad
        self.imageinfo = imageinfo
        self.findings = findings
        self.cancerinfo = cancerinfo
        self.validate()

    def validate(self):
        instanceofcheck(self.lungcad, LungCad)
        self.lungcad.validate()
        instanceofcheck(self.imageinfo, ImageInfo)
        self.imageinfo.validate()
        instanceofcheck(self.findings, (list, tuple))
        for finding in self.findings:
            instanceofcheck(finding, Finding)
            finding.validate()
        if self.cancerinfo is not None:
            instanceofcheck(self.cancerinfo, CancerInfo)
            self.cancerinfo.validate(findings=self.findings)

    def is_similar(
        self,
        other: XMLGeneratable,
        atol: float = 1e-8,
        different_runs: bool = False,
        compare_revision: bool = True,
    ) -> bool:
        if not isinstance(other, LungCadReport):
            return False
        if not self.lungcad.is_similar(
            other.lungcad,
            atol=atol,
            different_runs=different_runs,
            compare_revision=compare_revision,
        ):
            return False
        if not self.imageinfo.is_similar(other.imageinfo, atol=atol):
            return False
        if not optional_test(
            self.cancerinfo,
            other.cancerinfo,
            lambda a, b: a.is_similar(b, atol=atol),
        ):
            return False
        if len(self.findings) != len(other.findings):
            return False
        for f1, f2 in zip(self.findings, other.findings):
            if not f1.is_similar(f2, atol=atol):
                return False
        return True

    def to_xml(self) -> ElementTree.Element:
        self.validate()
        root = ElementTree.Element("LungCADReport")
        root.append(self.lungcad.to_xml())
        root.append(self.imageinfo.to_xml())
        if self.cancerinfo is not None:
            root.append(self.cancerinfo.to_xml())
        findings = ElementTree.SubElement(root, "Findings")
        for finding in self.findings:
            findings.append(finding.to_xml())
        return root

    def to_json(self) -> Dict:
        self.validate()
        result = dict(
            lungcad=self.lungcad.to_json(),
            imageinfo=self.imageinfo.to_json(),
            findings=[finding.to_json() for finding in self.findings],
        )
        if self.cancerinfo is not None:
            result["cancerinfo"] = self.cancerinfo.to_json()
        return result

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "LungCadReport":
        lungcad = LungCad.from_xml(none_guard_element(xml, "LungCAD"))
        imageinfo = ImageInfo.from_xml(none_guard_element(xml, "ImageInfo"))
        cancerxmlinfo = xml.find("./CancerInfo")
        cancerinfo = (
            None
            if cancerxmlinfo is None
            else CancerInfo.from_xml(cancerxmlinfo)
        )
        findings = []
        for child in xml.findall("./Findings/Finding"):
            findings.append(Finding.from_xml(child))
        return LungCadReport(
            lungcad=lungcad,
            imageinfo=imageinfo,
            findings=findings,
            cancerinfo=cancerinfo,
        )

    @staticmethod
    def from_json(data: Dict) -> "LungCadReport":
        lungcad = LungCad.from_json(data["lungcad"])
        imageinfo = ImageInfo.from_json(data["imageinfo"])
        cancerinfo = (
            None
            if not "cancerinfo" in data
            else CancerInfo.from_json(data["cancerinfo"])
        )
        findings = [Finding.from_json(finding) for finding in data["findings"]]
        return LungCadReport(
            lungcad, imageinfo, findings, cancerinfo=cancerinfo
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.lungcad == other.lungcad
            and self.imageinfo == other.imageinfo
            and self.cancerinfo == other.cancerinfo
            and len(self.findings) == len(other.findings)
            and all(
                [f1 == f2 for f1, f2 in zip(self.findings, other.findings)]
            )
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"lungcad:{self.lungcad}\nimageinfo:{self.imageinfo}\n"
            f"cancerinfo:{self.cancerinfo}\nfindings:{self.findings}"
        )


class LungCad(XMLGeneratable):
    def __init__(
        self,
        revision: str,
        name: str,
        datetimeofexecution: Union[str, datetime],
        trainingset1: str,
        trainingset2: str,
        coordinatesystem: str,
        computationtimeinseconds: Union[int, float],
    ):
        self.revision = revision
        self.name = name
        self.datetimeofexecution = datetimeofexecution
        self.trainingset1 = trainingset1
        self.trainingset2 = trainingset2
        self.coordinatesystem = coordinatesystem
        self.computationtimeinseconds = computationtimeinseconds

    @property
    def datetimeofexecution(self) -> Union[str, datetime]:
        return self.__datetimeofexecution

    @datetimeofexecution.setter
    def datetimeofexecution(self, datetimeofexecution: Union[str, datetime]):
        if isinstance(datetimeofexecution, datetime):
            self.__datetimeofexecution = datetimeofexecution.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            self.__datetimeofexecution = datetimeofexecution

    def validate(self):
        for attr in [
            "revision",
            "name",
            "trainingset1",
            "trainingset2",
            "coordinatesystem",
        ]:
            instanceofcheck(getattr(self, attr), str)
        if self.datetimeofexecution != "":
            matchcheck(
                self.datetimeofexecution,
                r"^(\d\d\d\d-(\d\d|[A-Z][a-z]{2})-\d\d|\d\d/\d\d/\d\d\d\d)\s\d\d:\d\d:\d\d(\.\d\d\d\d\d\d)?$",
            )
        if self.revision != "":
            matchcheck(self.revision.lower(), r"^[0-9a-f]{40}$")
        matchcheck(self.coordinatesystem.lower(), r"^world$")
        instanceofcheck(self.computationtimeinseconds, (int, float))

    def is_similar(
        self,
        other: object,
        atol: float = 1e-8,
        different_runs: bool = False,
        compare_revision: bool = True,
    ) -> bool:
        if not isinstance(other, LungCad):
            return False
        attrs = [
            "name",
            "trainingset1",
            "trainingset2",
            "coordinatesystem",
        ]
        if not different_runs:
            attrs.extend(["computationtimeinseconds", "datetimeofexecution"])
        if compare_revision:
            attrs.extend(["revision"])
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def to_xml(self) -> ElementTree.Element:
        info = ElementTree.Element("LungCAD")
        for attr in [
            "Revision",
            "Name",
            "DateTimeOfExecution",
            "TrainingSet1",
            "TrainingSet2",
            "CoordinateSystem",
            "ComputationTimeInSeconds",
        ]:
            ElementTree.SubElement(info, attr).text = str(
                getattr(self, attr.lower())
            )
        return info

    def to_json(self) -> Dict:
        return dict(
            revision=self.revision,
            name=self.name,
            datetimeofexecution=self.datetimeofexecution,
            trainingset1=self.trainingset1,
            trainingset2=self.trainingset2,
            coordinatesystem=self.coordinatesystem,
            computationtimeinseconds=self.computationtimeinseconds,
        )

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "LungCad":
        revision = get_xml_value(xml=xml, name="Revision", required=False)
        name = get_xml_value(xml=xml, name="Name", required=False)
        datetimeofexecution = get_xml_value(
            xml=xml, name="DateTimeOfExecution", required=False
        )
        trainingset1 = get_xml_value(
            xml=xml, name="TrainingSet1", required=False
        )
        trainingset2 = get_xml_value(
            xml=xml, name="TrainingSet2", required=False
        )
        coordinatesystem = get_xml_value(
            xml=xml, name="CoordinateSystem", required=True
        )
        computationtimeinseconds = float(
            get_xml_value(
                xml=xml, name="ComputationTimeInSeconds", required=True
            )
        )
        return LungCad(
            revision,
            name,
            datetimeofexecution,
            trainingset1,
            trainingset2,
            coordinatesystem,
            computationtimeinseconds,
        )

    @staticmethod
    def from_json(data: Dict) -> "LungCad":
        return LungCad(**data)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.revision == other.revision
            and self.name == other.name
            and self.datetimeofexecution == other.datetimeofexecution
            and self.trainingset1 == other.trainingset1
            and self.trainingset2 == other.trainingset2
            and self.coordinatesystem == other.coordinatesystem
            and self.computationtimeinseconds == other.computationtimeinseconds
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"revision:{self.revision} name:{self.name} datetimeofexecution:{self.datetimeofexecution} "
            f"trainingset1:{self.trainingset1} trainingset2:{self.trainingset2} "
            f"coordinatesystem:{self.coordinatesystem} computationtimeinseconds:{self.computationtimeinseconds}"
        )


def _make_tuple_or_none(
    e: Optional[
        Union[List[float], List[int], Tuple[float, ...], Tuple[int, ...]]
    ]
) -> Optional[Union[Tuple[float, ...], Tuple[int, ...]]]:
    if e is None:
        return None
    return tuple(e)


class Finding(XMLGeneratable):
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        z: float,
        probability: float,
        diameter_mm: float,
        volume_mm3: float,
        extent: Optional[Tuple[float, ...]] = None,
        cancerprobability: Optional[float] = None,
        noduletype: Optional[str] = None,
        nodulesegmentationseedvectorbegin: Optional[Tuple[float, ...]] = None,
        nodulesegmentationseedvectorend: Optional[Tuple[float, ...]] = None,
        solidcoresegmentationseedvectorbegin: Optional[
            Tuple[float, ...]
        ] = None,
        solidcoresegmentationseedvectorend: Optional[Tuple[float, ...]] = None,
        minoraxis_mm: Optional[float] = None,
        majoraxis_mm: Optional[float] = None,
        mass_mg: Optional[float] = None,
        corediameter_mm: Optional[float] = None,
        corevolume_mm3: Optional[float] = None,
        coremass_mg: Optional[float] = None,
        coremajoraxis_mm: Optional[float] = None,
        coreminoraxis_mm: Optional[float] = None,
        meandensity_hu: Optional[float] = None,
        coremeandensity_hu: Optional[float] = None,
        averagediameter_mm: Optional[float] = None,
        coreaveragediameter_mm: Optional[float] = None,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.probability = probability
        self.diameter_mm = diameter_mm
        self.volume_mm3 = volume_mm3
        self.extent = extent
        self.noduletype = noduletype
        self.nodulesegmentationseedvectorbegin = (
            nodulesegmentationseedvectorbegin
        )
        self.nodulesegmentationseedvectorend = nodulesegmentationseedvectorend
        self.solidcoresegmentationseedvectorbegin = (
            solidcoresegmentationseedvectorbegin
        )
        self.solidcoresegmentationseedvectorend = (
            solidcoresegmentationseedvectorend
        )
        self.minoraxis_mm = minoraxis_mm
        self.majoraxis_mm = majoraxis_mm
        self.mass_mg = mass_mg
        self.cancerprobability = cancerprobability
        self.corediameter_mm = corediameter_mm
        self.corevolume_mm3 = corevolume_mm3
        self.coremass_mg = coremass_mg
        self.coremajoraxis_mm = coremajoraxis_mm
        self.coreminoraxis_mm = coreminoraxis_mm
        self.meandensity_hu = meandensity_hu
        self.coremeandensity_hu = coremeandensity_hu

        self.averagediameter_mm = averagediameter_mm
        self.coreaveragediameter_mm = coreaveragediameter_mm

    @property
    def extent(self) -> Optional[Tuple[float, ...]]:
        return self.__extent

    @extent.setter
    def extent(self, extent: Optional[Tuple[float, ...]]):
        self.__extent = _make_tuple_or_none(extent)

    @property
    def nodulesegmentationseedvectorbegin(self) -> Optional[Tuple[float, ...]]:
        return self.__nodulesegmentationseedvectorbegin

    @nodulesegmentationseedvectorbegin.setter
    def nodulesegmentationseedvectorbegin(
        self, nodulesegmentationseedvectorbegin: Optional[Tuple[float, ...]]
    ):
        self.__nodulesegmentationseedvectorbegin = _make_tuple_or_none(
            nodulesegmentationseedvectorbegin
        )

    @property
    def nodulesegmentationseedvectorend(self) -> Optional[Tuple[float, ...]]:
        return self.__nodulesegmentationseedvectorend

    @nodulesegmentationseedvectorend.setter
    def nodulesegmentationseedvectorend(
        self, nodulesegmentationseedvectorend: Optional[Tuple[float, ...]]
    ):
        self.__nodulesegmentationseedvectorend = _make_tuple_or_none(
            nodulesegmentationseedvectorend
        )

    @property
    def solidcoresegmentationseedvectorbegin(
        self,
    ) -> Optional[Tuple[float, ...]]:
        return self.__solidcoresegmentationseedvectorbegin

    @solidcoresegmentationseedvectorbegin.setter
    def solidcoresegmentationseedvectorbegin(
        self, solidcoresegmentationseedvectorbegin: Optional[Tuple[float, ...]]
    ):
        self.__solidcoresegmentationseedvectorbegin = _make_tuple_or_none(
            solidcoresegmentationseedvectorbegin
        )

    @property
    def solidcoresegmentationseedvectorend(
        self,
    ) -> Optional[Tuple[float, ...]]:
        return self.__solidcoresegmentationseedvectorend

    @solidcoresegmentationseedvectorend.setter
    def solidcoresegmentationseedvectorend(
        self, solidcoresegmentationseedvectorend: Optional[Tuple[float, ...]]
    ):
        self.__solidcoresegmentationseedvectorend = _make_tuple_or_none(
            solidcoresegmentationseedvectorend
        )

    def validate(self):
        instanceofcheck(self.id, int)
        for var in [
            self.x,
            self.y,
            self.z,
            self.probability,
            self.diameter_mm,
            self.volume_mm3,
        ]:
            instanceofcheck(var, (int, float))
        for var in [
            self.cancerprobability,
            self.corediameter_mm,
            self.corevolume_mm3,
            self.coremass_mg,
            self.coremajoraxis_mm,
            self.coreminoraxis_mm,
            self.coremeandensity_hu,
            self.majoraxis_mm,
            self.minoraxis_mm,
            self.mass_mg,
            self.meandensity_hu,
            self.averagediameter_mm,
            self.coreaveragediameter_mm,
        ]:
            if var is not None:
                instanceofcheck(var, (int, float))
        for var in [
            self.extent,
            self.nodulesegmentationseedvectorbegin,
            self.nodulesegmentationseedvectorend,
            self.solidcoresegmentationseedvectorbegin,
            self.solidcoresegmentationseedvectorend,
        ]:
            if var is not None:
                lencheck(var, 3)
                for element in var:
                    instanceofcheck(element, (int, float))
        if (self.nodulesegmentationseedvectorbegin is None) ^ (
            self.nodulesegmentationseedvectorend is None
        ):
            raise ValueError(
                "nodulesegmentationseedvectorbegin and nodulesegmentationseedvectorend must both None or both set."
            )
        if (self.solidcoresegmentationseedvectorbegin is None) ^ (
            self.solidcoresegmentationseedvectorend is None
        ):
            raise ValueError(
                "solidcoresegmentationseedvectorbegin and solidcoresegmentationseedvectorend must both None or both set."
            )
        if (
            (self.solidcoresegmentationseedvectorbegin is not None)
            and (self.solidcoresegmentationseedvectorend is not None)
            and self.noduletype != "PartSolid"
        ):
            raise ValueError(
                "When solidcoresegmentationseedvectors are set, noduletype is expected to be set to `PartSolid`."
            )
        if self.noduletype is not None:
            instanceofcheck(self.noduletype, str)

    def is_similar(self, other: object, atol: float = 1e-8) -> bool:
        if not isinstance(other, Finding):
            return False
        attrs = [
            "id",
            "x",
            "y",
            "z",
            "diameter_mm",
            "volume_mm3",
            "probability",
        ]
        for attr in attrs:
            if not np.isclose(
                getattr(self, attr), getattr(other, attr), atol=atol
            ):
                return False
        for attr in [
            "cancerprobability",
            "corediameter_mm",
            "corevolume_mm3",
            "coremass_mg",
            "coremajoraxis_mm",
            "coreminoraxis_mm",
            "majoraxis_mm",
            "minoraxis_mm",
            "mass_mg",
            "meandensity_hu",
            "coremeandensity_hu",
            "averagediameter_mm",
            "coreaveragediameter_mm",
        ]:
            if not optional_test(
                getattr(self, attr),
                getattr(other, attr),
                lambda a, b: np.allclose(a, b, atol=atol),
            ):
                return False
        for attr in [
            "extent",
            "nodulesegmentationseedvectorbegin",
            "nodulesegmentationseedvectorend",
            "solidcoresegmentationseedvectorbegin",
            "solidcoresegmentationseedvectorend",
        ]:
            if not optional_test(
                getattr(self, attr),
                getattr(other, attr),
                lambda a, b: len(a) == len(b) and np.allclose(a, b, atol=atol),
            ):
                return False
        if self.noduletype != other.noduletype:
            return False
        return True

    def to_xml(self) -> ElementTree.Element:
        finding = ElementTree.Element("Finding")
        for attr in [
            "ID",
            "X",
            "Y",
            "Z",
            "Extent",
            "Probability",
            "CancerProbability",
            "Diameter_mm",
            "Volume_mm3",
            "NoduleType",
            "NoduleSegmentationSeedVectorBegin",
            "NoduleSegmentationSeedVectorEnd",
            "SolidCoreSegmentationSeedVectorBegin",
            "SolidCoreSegmentationSeedVectorEnd",
            "MinorAxis_mm",
            "MajorAxis_mm",
            "Mass_mg",
            "MeanDensity_HU",
            "CoreDiameter_mm",
            "CoreVolume_mm3",
            "CoreMass_mg",
            "CoreMajorAxis_mm",
            "CoreMinorAxis_mm",
            "CoreMeanDensity_HU",
            "AverageDiameter_mm",
            "CoreAverageDiameter_mm",
        ]:
            if attr == "NoduleType":
                if self.noduletype is not None:
                    ElementTree.SubElement(
                        finding, attr
                    ).text = self.noduletype
            elif attr == "Extent":
                if self.extent is not None:
                    ext = ElementTree.SubElement(finding, "Extent")
                    for e, idx in zip(self.extent, ["X", "Y", "Z"]):
                        ElementTree.SubElement(ext, f"Extent{idx}").text = str(
                            e
                        )
            elif attr in [
                "NoduleSegmentationSeedVectorBegin",
                "NoduleSegmentationSeedVectorEnd",
                "SolidCoreSegmentationSeedVectorBegin",
                "SolidCoreSegmentationSeedVectorEnd",
            ]:
                value = getattr(self, attr.lower())
                if value is not None:
                    ElementTree.SubElement(finding, attr).text = ",".join(
                        map(str, value)
                    )
            else:
                value = getattr(self, attr.lower())
                if attr in [
                    "MinorAxis_mm",
                    "MajorAxis_mm",
                    "Mass_mg",
                    "MeanDensity_HU",
                    "CancerProbability",
                    "CoreDiameter_mm",
                    "CoreVolume_mm3",
                    "CoreMass_mg",
                    "CoreMajorAxis_mm",
                    "CoreMinorAxis_mm",
                    "CoreMeanDensity_HU",
                    "AverageDiameter_mm",
                    "CoreAverageDiameter_mm",
                ]:
                    if value is not None:
                        ElementTree.SubElement(finding, attr).text = str(value)
                else:
                    ElementTree.SubElement(finding, attr).text = str(value)
        return finding

    def to_json(self) -> Dict:
        return dict(
            id=self.id,
            x=self.x,
            y=self.y,
            z=self.z,
            probability=self.probability,
            diameter_mm=self.diameter_mm,
            volume_mm3=self.volume_mm3,
            extent=self.extent,
            noduletype=self.noduletype,
            nodulesegmentationseedvectorbegin=self.nodulesegmentationseedvectorbegin,
            nodulesegmentationseedvectorend=self.nodulesegmentationseedvectorend,
            solidcoresegmentationseedvectorbegin=self.solidcoresegmentationseedvectorbegin,
            solidcoresegmentationseedvectorend=self.solidcoresegmentationseedvectorend,
            majoraxis_mm=self.majoraxis_mm,
            minoraxis_mm=self.minoraxis_mm,
            mass_mg=self.mass_mg,
            meandensity_hu=self.meandensity_hu,
            cancerprobability=self.cancerprobability,
            corediameter_mm=self.corediameter_mm,
            corevolume_mm3=self.corevolume_mm3,
            coremass_mg=self.coremass_mg,
            coremajoraxis_mm=self.coremajoraxis_mm,
            coreminoraxis_mm=self.coreminoraxis_mm,
            coremeandensity_hu=self.coremeandensity_hu,
            averagediameter_mm=self.averagediameter_mm,
            coreaveragediameter_mm=self.coreaveragediameter_mm,
        )

    @staticmethod
    def from_json(data: Dict) -> "Finding":
        return Finding(**data)

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "Finding":
        extentxml = xml.find("./Extent")
        extent = (
            None
            if extentxml is None
            else tuple(
                [
                    float(
                        get_xml_value(
                            xml=xml,
                            name=f"Extent/Extent{element}",
                            required=True,
                        )
                    )
                    for element in ["X", "Y", "Z"]
                ]
            )
        )

        def find_and_format_optional_vector(
            name: str,
        ) -> Optional[Union[Tuple[int, ...], Tuple[float, ...]]]:
            vector = get_xml_value(xml=xml, name=name, required=False)
            if vector == "":
                return None
            return tuple(map(float, vector.split(",")))

        nodulesegmentationseedvectorbegin = find_and_format_optional_vector(
            "NoduleSegmentationSeedVectorBegin"
        )
        nodulesegmentationseedvectorend = find_and_format_optional_vector(
            "NoduleSegmentationSeedVectorEnd"
        )
        solidcoresegmentationseedvectorbegin = find_and_format_optional_vector(
            "SolidCoreSegmentationSeedVectorBegin"
        )
        solidcoresegmentationseedvectorend = find_and_format_optional_vector(
            "SolidCoreSegmentationSeedVectorEnd"
        )
        cancerprobability = get_xml_value_or_none(
            xml=xml, name="CancerProbability"
        )
        noduletype = get_xml_value_or_none(xml=xml, name="NoduleType")
        majoraxis_mm = get_xml_value_or_none(xml=xml, name="MajorAxis_mm")
        minoraxis_mm = get_xml_value_or_none(xml=xml, name="MinorAxis_mm")
        mass_mg = get_xml_value_or_none(xml=xml, name="Mass_mg")
        meandensity_hu = get_xml_value_or_none(xml=xml, name="MeanDensity_HU")
        corediameter_mm = get_xml_value_or_none(
            xml=xml, name="CoreDiameter_mm"
        )
        corevolume_mm3 = get_xml_value_or_none(xml=xml, name="CoreVolume_mm3")
        coremass_mg = get_xml_value_or_none(xml=xml, name="CoreMass_mg")
        coremajoraxis_mm = get_xml_value_or_none(
            xml=xml, name="CoreMajorAxis_mm"
        )
        coreminoraxis_mm = get_xml_value_or_none(
            xml=xml, name="CoreMinorAxis_mm"
        )
        coremeandensity_hu = get_xml_value_or_none(
            xml=xml, name="CoreMeanDensity_HU"
        )

        averagediameter_mm = get_xml_value_or_none(
            xml=xml, name="AverageDiameter_mm"
        )
        coreaveragediameter_mm = get_xml_value_or_none(
            xml=xml, name="CoreAverageDiameter_mm"
        )

        return Finding(
            id=int(get_xml_value(xml=xml, name="ID", required=True)),
            x=float(get_xml_value(xml=xml, name="X", required=True)),
            y=float(get_xml_value(xml=xml, name="Y", required=True)),
            z=float(get_xml_value(xml=xml, name="Z", required=True)),
            probability=float(
                get_xml_value(xml=xml, name="Probability", required=True)
            ),
            diameter_mm=float(
                get_xml_value(xml=xml, name="Diameter_mm", required=True)
            ),
            volume_mm3=float(
                get_xml_value(xml=xml, name="Volume_mm3", required=True)
            ),
            cancerprobability=None
            if cancerprobability is None
            else float(cancerprobability),
            extent=extent,
            noduletype=noduletype,
            nodulesegmentationseedvectorbegin=nodulesegmentationseedvectorbegin,
            nodulesegmentationseedvectorend=nodulesegmentationseedvectorend,
            solidcoresegmentationseedvectorbegin=solidcoresegmentationseedvectorbegin,
            solidcoresegmentationseedvectorend=solidcoresegmentationseedvectorend,
            majoraxis_mm=None if majoraxis_mm is None else float(majoraxis_mm),
            minoraxis_mm=None if minoraxis_mm is None else float(minoraxis_mm),
            mass_mg=None if mass_mg is None else float(mass_mg),
            meandensity_hu=None
            if meandensity_hu is None
            else float(meandensity_hu),
            corediameter_mm=None
            if corediameter_mm is None
            else float(corediameter_mm),
            corevolume_mm3=None
            if corevolume_mm3 is None
            else float(corevolume_mm3),
            coremass_mg=None if coremass_mg is None else float(coremass_mg),
            coremajoraxis_mm=None
            if coremajoraxis_mm is None
            else float(coremajoraxis_mm),
            coreminoraxis_mm=None
            if coreminoraxis_mm is None
            else float(coreminoraxis_mm),
            coremeandensity_hu=None
            if coremeandensity_hu is None
            else float(coremeandensity_hu),
            averagediameter_mm=None
            if averagediameter_mm is None
            else float(averagediameter_mm),
            coreaveragediameter_mm=None
            if coreaveragediameter_mm is None
            else float(coreaveragediameter_mm),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.id == other.id
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.extent == other.extent
            and self.probability == other.probability
            and self.cancerprobability == other.cancerprobability
            and self.diameter_mm == other.diameter_mm
            and self.volume_mm3 == other.volume_mm3
            and self.noduletype == other.noduletype
            and self.nodulesegmentationseedvectorbegin
            == other.nodulesegmentationseedvectorbegin
            and self.nodulesegmentationseedvectorend
            == other.nodulesegmentationseedvectorend
            and self.solidcoresegmentationseedvectorbegin
            == other.solidcoresegmentationseedvectorbegin
            and self.solidcoresegmentationseedvectorend
            == other.solidcoresegmentationseedvectorend
            and self.minoraxis_mm == other.minoraxis_mm
            and self.majoraxis_mm == other.majoraxis_mm
            and self.mass_mg == other.mass_mg
            and self.meandensity_hu == other.meandensity_hu
            and self.corediameter_mm == other.corediameter_mm
            and self.corevolume_mm3 == other.corevolume_mm3
            and self.coremass_mg == other.coremass_mg
            and self.coremajoraxis_mm == other.coremajoraxis_mm
            and self.coreminoraxis_mm == other.coreminoraxis_mm
            and self.coremeandensity_hu == other.coremeandensity_hu
            and self.averagediameter_mm == other.averagediameter_mm
            and self.coreaveragediameter_mm == other.coreaveragediameter_mm
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"id:{self.id} x:{self.x} y:{self.y} z:{self.z} extent:{self.extent} probability:{self.probability} "
            f"cancerprobability:{self.cancerprobability} diameter_mm:{self.diameter_mm} volume_mm3:{self.volume_mm3} noduletype:{self.noduletype} "
            f"nodulesegmentationseedvectorbegin:{self.nodulesegmentationseedvectorbegin} nodulesegmentationseedvectorend:{self.nodulesegmentationseedvectorend} "
            f"majoraxis_mm:{self.majoraxis_mm} minoraxis_mm:{self.minoraxis_mm} mass_mg:{self.mass_mg} meandensity_hu:{self.meandensity_hu} "
            f"solidcoresegmentationseedvectorbegin:{self.solidcoresegmentationseedvectorbegin} solidcoresegmentationseedvectorend:{self.solidcoresegmentationseedvectorend} "
            f"corediameter_mm:{self.corediameter_mm} corevolume_mm3:{self.corevolume_mm3} coremass_mg:{self.coremass_mg} "
            f"coremajoraxis_mm:{self.coremajoraxis_mm} coreminoraxis_mm:{self.coreminoraxis_mm} coremeandensity_hu:{self.coremeandensity_hu}"
            f"averagediameter_mm:{self.averagediameter_mm} coreaveragediameter_mm:{self.coreaveragediameter_mm}"
        )


class ImageInfo(XMLGeneratable):
    def __init__(
        self,
        dimensions: Tuple[int, ...],
        voxelsize: Tuple[float, ...],
        origin: Tuple[float, ...],
        orientation: Tuple[float, ...],
        patientuid: str,
        studyuid: str,
        seriesuid: str,
    ):
        self.dimensions = dimensions
        self.voxelsize = voxelsize
        self.origin = origin
        self.orientation = orientation
        self.patientuid = patientuid
        self.studyuid = studyuid
        self.seriesuid = seriesuid

    @property
    def dimensions(self) -> Tuple[int, ...]:
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Tuple[int, ...]):
        self.__dimensions = tuple(dimensions)

    @property
    def origin(self) -> Tuple[float, ...]:
        return self.__origin

    @origin.setter
    def origin(self, origin: Tuple[float, ...]):
        self.__origin = tuple(origin)

    @property
    def orientation(self) -> Tuple[float, ...]:
        return self.__orientation

    @orientation.setter
    def orientation(self, orientation: Tuple[float, ...]):
        self.__orientation = tuple(orientation)

    @property
    def voxelsize(self) -> Tuple[float, ...]:
        return self.__voxelsize

    @voxelsize.setter
    def voxelsize(self, voxelsize: Tuple[float, ...]):
        self.__voxelsize = tuple(voxelsize)

    def validate(self):
        for var, typ in (
            (self.dimensions, int),
            (self.voxelsize, (int, float)),
            (self.origin, (int, float)),
        ):
            lencheck(var, 3)
            for element in var:
                instanceofcheck(element, typ)
        lencheck(self.orientation, 9)
        for element in self.orientation:
            instanceofcheck(element, (int, float))
        for var in (self.patientuid, self.studyuid):
            instanceofcheck(var, str)
            matchcheck(var, r"^([\d.]*|None)$")
        instanceofcheck(self.seriesuid, str)

    def is_similar(self, other: object, atol: float = 1e-8) -> bool:
        if not isinstance(other, ImageInfo):
            return False
        for attr in ["dimensions", "voxelsize", "orientation", "origin"]:
            a, b = getattr(self, attr), getattr(other, attr)
            if not (len(a) == len(b) and np.allclose(a, b, atol=atol)):
                return False
        for attr in ["patientuid", "studyuid", "seriesuid"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        if tuple(self.dimensions) != tuple(other.dimensions):
            return False
        return True

    def to_xml(self) -> ElementTree.Element:
        info = ElementTree.Element("ImageInfo")
        for attr, subattr in (
            ("Dimensions", "dim"),
            ("VoxelSize", "voxelSize"),
            ("Origin", "origin"),
        ):
            sub = ElementTree.SubElement(info, attr)
            for idx, element in enumerate(["X", "Y", "Z"]):
                ElementTree.SubElement(sub, f"{subattr}{element}").text = str(
                    getattr(self, attr.lower())[idx]
                )
        ElementTree.SubElement(info, "Orientation").text = ",".join(
            [str(e) for e in self.orientation]
        )
        for attr in ["PatientUID", "StudyUID", "SeriesUID"]:
            ElementTree.SubElement(info, attr).text = str(
                getattr(self, attr.lower())
            )
        return info

    def to_json(self) -> Dict:
        return dict(
            dimensions=self.dimensions,
            voxelsize=self.voxelsize,
            origin=self.origin,
            orientation=self.orientation,
            patientuid=self.patientuid,
            studyuid=self.studyuid,
            seriesuid=self.seriesuid,
        )

    @staticmethod
    def from_json(data: Dict) -> "ImageInfo":
        return ImageInfo(**data)

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "ImageInfo":
        def get_elements(
            xml: Union[ElementTree.Element, ElementTree.ElementTree],
            attr: str,
            subattr: str,
            typ: Callable,
        ) -> Tuple[Any, ...]:
            elements = tuple(
                [
                    typ(
                        get_xml_value(
                            xml=xml,
                            name=f"{attr}/{subattr}{element}",
                            required=True,
                        )
                    )
                    for element in ["X", "Y", "Z"]
                ]
            )
            return elements

        dimensions = get_elements(
            xml=xml, attr="Dimensions", subattr="dim", typ=int
        )
        voxelsize = get_elements(
            xml=xml, attr="VoxelSize", subattr="voxelSize", typ=float
        )
        origin = get_elements(
            xml=xml, attr="Origin", subattr="origin", typ=float
        )
        orientation = tuple(
            map(
                float,
                get_xml_value(
                    xml=xml, name="Orientation", required=True
                ).split(","),
            )
        )
        patientuid = get_xml_value(xml=xml, name="PatientUID", required=False)
        studyuid = get_xml_value(xml=xml, name="StudyUID", required=False)
        seriesuid = get_xml_value(xml=xml, name="SeriesUID", required=True)
        patientuid = patientuid if patientuid is not None else ""
        return ImageInfo(
            dimensions=dimensions,
            voxelsize=voxelsize,
            origin=origin,
            orientation=orientation,
            patientuid=patientuid,
            studyuid=studyuid,
            seriesuid=seriesuid,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.dimensions == other.dimensions
            and self.voxelsize == other.voxelsize
            and self.origin == other.origin
            and self.orientation == other.orientation
            and self.patientuid == other.patientuid
            and self.studyuid == other.studyuid
            and self.seriesuid == other.seriesuid
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"dimensions:{self.dimensions} voxelsize:{self.voxelsize} origin:{self.origin} orientation:{self.orientation} "
            f"patientuid:{self.patientuid} studyuid:{self.studyuid} seriesuid:{self.seriesuid}"
        )


class CancerInfo(XMLGeneratable):
    def __init__(self, casecancerprobability, referencenoduleids):
        self.casecancerprobability = casecancerprobability
        self.referencenoduleids = referencenoduleids

    @property
    def referencenoduleids(self) -> Tuple[int, ...]:
        return self.__referencenoduleids

    @referencenoduleids.setter
    def referencenoduleids(
        self, referencenoduleids: Union[List[int], Tuple[int, ...]]
    ):
        self.__referencenoduleids = tuple(referencenoduleids)

    def validate(self, findings: Optional[List[Finding]] = None):
        instanceofcheck(self.casecancerprobability, (int, float))
        instanceofcheck(self.referencenoduleids, (list, tuple))
        for e in self.referencenoduleids:
            instanceofcheck(e, int)
        if findings is not None:
            findings_ids = set([finding.id for finding in findings])
            for e in self.referencenoduleids:
                if e not in findings_ids:
                    raise ValueError(
                        f"Reference nodule id {e} was not found " "in findings"
                    )

    def is_similar(self, other, atol=1e-8):
        if not isinstance(other, CancerInfo):
            return False
        if not np.isclose(
            self.casecancerprobability, other.casecancerprobability
        ):
            return False
        if tuple(self.referencenoduleids) != tuple(other.referencenoduleids):
            return False
        return True

    def to_xml(self):
        info = ElementTree.Element("CancerInfo")
        ElementTree.SubElement(info, "CaseCancerProbability").text = str(
            self.casecancerprobability
        )
        ElementTree.SubElement(info, "ReferenceNoduleIDs").text = ",".join(
            map(str, self.referencenoduleids)
        )
        return info

    def to_json(self) -> Dict:
        return dict(
            casecancerprobability=self.casecancerprobability,
            referencenoduleids=self.referencenoduleids,
        )

    @staticmethod
    def from_json(data: Dict) -> "CancerInfo":
        return CancerInfo(**data)

    @staticmethod
    def from_xml(
        xml: Union[ElementTree.Element, ElementTree.ElementTree]
    ) -> "CancerInfo":
        ref_nodule_ids = get_xml_value_or_none(
            xml=xml, name="ReferenceNoduleIDs"
        )
        referencenoduleids = (
            ()
            if ref_nodule_ids is None
            else tuple(map(int, ref_nodule_ids.split(",")))
        )
        casecancerprobability = float(
            get_xml_value(xml=xml, name="CaseCancerProbability", required=True)
        )
        return CancerInfo(
            casecancerprobability=casecancerprobability,
            referencenoduleids=referencenoduleids,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.referencenoduleids == other.referencenoduleids
            and self.casecancerprobability == other.casecancerprobability
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"casecancerprobability:{self.casecancerprobability} referencenoduleids:{self.referencenoduleids}"