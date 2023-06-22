from pathlib import Path
import xml.etree.ElementTree as et
from datetime import datetime
from xml.etree import ElementTree
from xml.dom import minidom
import re
from abc import ABCMeta, abstractmethod
import subprocess as sp


def get_current_git_hash(git_dir: Path = Path(__file__).parent / ".git") -> str:
    head_file = git_dir / "HEAD"
    with open(str(head_file), "r") as f:
        content = f.read().strip().split(" ")
    if "ref:" != content[0]:
        return content[0]
    with open(str(git_dir / Path(content[1])), "r") as f:
        git_hash = f.read().strip()
    return git_hash


class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


def instanceofcheck(obj, typ):
    if not isinstance(obj, typ):
        raise ValueError(
            "Object type ({}) is not of type: {}".format(type(obj), typ)
        )


def lencheck(obj, length):
    instanceofcheck(obj, (list, tuple))
    if not len(obj) == length:
        raise ValueError(
            "Object length is ({}), expected: {}".format(len(obj), length)
        )


def matchcheck(obj, pattern):
    if not re.match(pattern, obj.strip()):
        raise ValueError(
            "{} does not match regex. pattern: {}".format(obj, pattern)
        )


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def none_str_guard(obj):
    if obj is None:
        return ""
    elif obj.lower() == "unknown":
        return ""
    return obj


class XMLGeneratable(object, metaclass=ABCMeta):
    @abstractmethod
    def xml_element(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractstatic
    def from_xml(xml):
        raise NotImplementedError()

    @classmethod
    def from_file(cls, fname):
        return cls.from_xml(ElementTree.parse(str(fname)))

    def to_file(self, fname):
        with open(str(fname), "wb") as f:
            ElementTree.ElementTree(self.xml_element()).write(
                f, encoding="UTF-8", xml_declaration=True
            )


class LungCadReport(XMLGeneratable):
    def __init__(self, lungcad, imageinfo, findings, cancerinfo=None):
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

    def xml_element(self):
        self.validate()
        root = et.Element("LungCADReport")
        root.append(self.lungcad.xml_element())
        root.append(self.imageinfo.xml_element())
        if self.cancerinfo is not None:
            root.append(self.cancerinfo.xml_element())
        findings = et.SubElement(root, "Findings")
        for finding in self.findings:
            findings.append(finding.xml_element())
        return root

    @staticmethod
    def from_xml(xml):
        lungcad = LungCad.from_xml(xml.find("./LungCAD"))
        imageinfo = ImageInfo.from_xml(xml.find("./ImageInfo"))
        cancerinfo = xml.find("./CancerInfo")
        if cancerinfo is not None:
            cancerinfo = CancerInfo.from_xml(cancerinfo)
        findings = []
        for child in xml.findall("./Findings/Finding"):
            findings.append(Finding.from_xml(child))
        return LungCadReport(
            lungcad, imageinfo, findings, cancerinfo=cancerinfo
        )

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "LungCAD: {}\nImageInfo: {}\nCancerInfo: {}\nFindings:{}".format(
            self.lungcad, self.imageinfo, self.cancerinfo, self.findings
        )


class LungCad(XMLGeneratable):
    def __init__(
        self,
        revision,
        name,
        datetimeofexecution,
        trainingset1,
        trainingset2,
        coordinatesystem,
        computationtimeinseconds,
    ):
        self.revision = revision
        self.name = name
        self.datetimeofexecution = (
            datetimeofexecution.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(datetimeofexecution, datetime)
            else datetimeofexecution
        )
        self.trainingset1 = trainingset1
        self.trainingset2 = trainingset2
        self.coordinatesystem = coordinatesystem
        self.computationtimeinseconds = computationtimeinseconds

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

    def xml_element(self):
        info = et.Element("LungCAD")
        for attr in [
            "Revision",
            "Name",
            "DateTimeOfExecution",
            "TrainingSet1",
            "TrainingSet2",
            "CoordinateSystem",
            "ComputationTimeInSeconds",
        ]:
            et.SubElement(info, attr).text = str(getattr(self, attr.lower()))
        return info

    @staticmethod
    def from_xml(xml):
        revision = none_str_guard(xml.find("./Revision").text)
        name = none_str_guard(xml.find("./Name").text)
        datetimeofexecution = none_str_guard(
            xml.find("./DateTimeOfExecution").text
        )
        trainingset1 = none_str_guard(xml.find("./TrainingSet1").text)
        trainingset2 = none_str_guard(xml.find("./TrainingSet2").text)
        coordinatesystem = xml.find("./CoordinateSystem").text
        computationtimeinseconds = float(
            xml.find("./ComputationTimeInSeconds").text
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

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "revision: {} name: {} datetimeofexecution: {} trainingset1: {} trainingset2: {} coordinatesystem: {} computationtimeinseconds: {}".format(
            self.revision,
            self.name,
            self.datetimeofexecution,
            self.trainingset1,
            self.trainingset2,
            self.coordinatesystem,
            self.computationtimeinseconds,
        )


class Finding(XMLGeneratable):
    def __init__(
        self,
        id,
        x,
        y,
        z,
        probability,
        diameter_mm,
        volume_mm3,
        extent=None,
        cancerprobability=None,
        noduletype=None,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.probability = probability
        self.diameter_mm = diameter_mm
        self.volume_mm3 = volume_mm3
        self.extent = extent
        if self.extent is not None:
            self.extent = tuple(extent)
        self.cancerprobability = cancerprobability
        self.noduletype = noduletype

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
        if self.cancerprobability is not None:
            instanceofcheck(self.cancerprobability, (int, float))
        if self.extent is not None:
            lencheck(self.extent, 3)
            for element in self.extent:
                instanceofcheck(element, (int, float))
        if self.noduletype is not None:
            instanceofcheck(self.noduletype, str)

    def xml_element(self):
        finding = et.Element("Finding")
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
        ]:
            if attr == "CancerProbability":
                if self.cancerprobability is not None:
                    et.SubElement(finding, attr).text = str(
                        self.cancerprobability
                    )
            elif attr == "NoduleType":
                if self.noduletype is not None:
                    et.SubElement(finding, attr).text = self.noduletype
            elif attr == "Extent":
                if self.extent is not None:
                    ext = et.SubElement(finding, "Extent")
                    for e, idx in zip(self.extent, ["X", "Y", "Z"]):
                        et.SubElement(ext, "Extent{}".format(idx)).text = str(
                            e
                        )
            else:
                et.SubElement(finding, attr).text = str(
                    getattr(self, attr.lower())
                )
        return finding

    @staticmethod
    def from_xml(xml):
        id = int(xml.find("./ID").text)
        x = float(xml.find("./X").text)
        y = float(xml.find("./Y").text)
        z = float(xml.find("./Z").text)
        probability = float(xml.find("./Probability").text)
        diameter_mm = float(xml.find("./Diameter_mm").text)
        volume_mm3 = float(xml.find("./Volume_mm3").text)
        cancerprobability = xml.find("./CancerProbability")
        if cancerprobability is not None:
            cancerprobability = float(cancerprobability.text)
        extent = xml.find("./Extent")
        if extent is not None:
            extent = tuple(
                [
                    float(xml.find("./Extent/Extent{}".format(element)).text)
                    for element in ["X", "Y", "Z"]
                ]
            )
        noduletype = xml.find("./NoduleType")
        if noduletype is not None:
            noduletype = noduletype.text
        return Finding(
            id,
            x,
            y,
            z,
            probability,
            diameter_mm,
            volume_mm3,
            cancerprobability=cancerprobability,
            extent=extent,
            noduletype=noduletype,
        )

    def __eq__(self, other):
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
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return (
            "id: {} x: {} y: {} z: {} extent: {} probability: {} "
            "cancerprobability: {} diameter_mm: {} volume_mm3: {}  noduletype: {}".format(
                self.id,
                self.x,
                self.y,
                self.z,
                self.extent,
                self.probability,
                self.cancerprobability,
                self.diameter_mm,
                self.volume_mm3,
                self.noduletype,
            )
        )


class ImageInfo(XMLGeneratable):
    def __init__(
        self,
        dimensions,
        voxelsize,
        origin,
        orientation,
        patientuid,
        studyuid,
        seriesuid,
    ):
        self.dimensions = dimensions
        self.voxelsize = voxelsize
        self.origin = origin
        self.orientation = orientation
        self.patientuid = patientuid
        self.studyuid = studyuid
        self.seriesuid = seriesuid

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

    def xml_element(self):
        info = et.Element("ImageInfo")
        for attr, subattr in (
            ("Dimensions", "dim"),
            ("VoxelSize", "voxelSize"),
            ("Origin", "origin"),
        ):
            sub = et.SubElement(info, attr)
            for idx, element in enumerate(["X", "Y", "Z"]):
                et.SubElement(sub, "{}{}".format(subattr, element)).text = str(
                    getattr(self, attr.lower())[idx]
                )
        et.SubElement(info, "Orientation").text = ",".join(
            [str(e) for e in self.orientation]
        )
        for attr in ["PatientUID", "StudyUID", "SeriesUID"]:
            et.SubElement(info, attr).text = str(getattr(self, attr.lower()))
        return info

    @staticmethod
    def from_xml(xml):
        t = {}
        for attr, subattr, typ in (
            ("Dimensions", "dim", int),
            ("VoxelSize", "voxelSize", float),
            ("Origin", "origin", float),
        ):
            t[attr.lower()] = tuple(
                [
                    typ(
                        xml.find(
                            "./{}/{}{}".format(attr, subattr, element)
                        ).text
                    )
                    for element in ["X", "Y", "Z"]
                ]
            )
        orientation = tuple(
            [float(e) for e in xml.find("./Orientation").text.split(",")]
        )
        patientuid = none_str_guard(xml.find("./PatientUID").text)
        studyuid = none_str_guard(xml.find("./StudyUID").text)
        seriesuid = xml.find("./SeriesUID").text
        patientuid = patientuid if patientuid is not None else ""
        return ImageInfo(
            t["dimensions"],
            t["voxelsize"],
            t["origin"],
            orientation,
            patientuid,
            studyuid,
            seriesuid,
        )

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return (
            "dimensions: {} voxelsize: {} origin: {} orientation: {} "
            "patientuid: {} studyuid: {} seriesuid: {}".format(
                self.dimensions,
                self.voxelsize,
                self.origin,
                self.orientation,
                self.patientuid,
                self.studyuid,
                self.seriesuid,
            )
        )


class CancerInfo(XMLGeneratable):
    def __init__(self, casecancerprobability, referencenoduleids):
        self.casecancerprobability = casecancerprobability
        self.referencenoduleids = tuple(referencenoduleids)

    def validate(self, findings=None):
        instanceofcheck(self.casecancerprobability, (int, float))
        instanceofcheck(self.referencenoduleids, (list, tuple))
        for e in self.referencenoduleids:
            instanceofcheck(e, int)
        if findings is not None:
            findings_ids = set([finding.id for finding in findings])
            for e in self.referencenoduleids:
                if e not in findings_ids:
                    raise ValueError(
                        "Reference nodule id {} was not found "
                        "in findings".format(e)
                    )

    def xml_element(self):
        info = et.Element("CancerInfo")
        et.SubElement(info, "CaseCancerProbability").text = str(
            self.casecancerprobability
        )
        et.SubElement(info, "ReferenceNoduleIDs").text = ",".join(
            [str(e) for e in self.referencenoduleids]
        )
        return info

    @staticmethod
    def from_xml(xml):
        ref_nodule_ids = xml.find("./ReferenceNoduleIDs").text
        referencenoduleids = ()
        if ref_nodule_ids is not None:
            referencenoduleids = tuple(
                [int(e) for e in ref_nodule_ids.split(",")]
            )
        casecancerprobability = float(xml.find("./CaseCancerProbability").text)
        return CancerInfo(
            casecancerprobability=casecancerprobability,
            referencenoduleids=referencenoduleids,
        )

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.referencenoduleids == other.referencenoduleids
            and self.casecancerprobability == other.casecancerprobability
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "casecancerprobability: {} referencenoduleids: {}".format(
            self.casecancerprobability, self.referencenoduleids
        )
