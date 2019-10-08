import xml.etree.ElementTree as et
from xml.etree import ElementTree
from xml.dom import minidom
import re
from abc import ABCMeta, abstractmethod


class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


def instanceofcheck(obj, typ):
    if not isinstance(obj, typ):
        raise ValueError("Object type ({}) is not of type: {}".format(type(obj), typ))


def lencheck(obj, length):
    instanceofcheck(obj, (list, tuple))
    if not len(obj) == length:
        raise ValueError("Object length is ({}), expected: {}".format(len(obj), length))


def matchcheck(obj, pattern):
    if not re.match(pattern, obj.strip()):
        raise ValueError("{} does not match regex. pattern: {}".format(obj, pattern))


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def none_str_guard(obj):
    if obj is None:
        return ""
    return obj


class XMLGeneratable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def xml_element(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractstatic
    def from_xml(xml):
        raise NotImplementedError()


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
            self.cancerinfo.validate()

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
        return LungCadReport(lungcad, imageinfo, findings, cancerinfo=cancerinfo)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               self.lungcad == other.lungcad and \
               self.imageinfo == other.imageinfo and \
               self.cancerinfo == other.cancerinfo and \
               len(self.findings) == len(other.findings) and \
               all([f1 == f2 for f1, f2 in zip(self.findings, other.findings)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "LungCAD: {}\nImageInfo: {}\nCancerInfo: {}\nFindings:{}".format(self.lungcad, self.imageinfo, self.cancerinfo, self.findings)


class LungCad(XMLGeneratable):
    def __init__(self, revision, name, datetimeofexecution, trainingset1, trainingset2,
                 coordinatesystem, computationtimeinseconds):
        self.revision = revision
        self.name = name
        self.datetimeofexecution = datetimeofexecution
        self.trainingset1 = trainingset1
        self.trainingset2 = trainingset2
        self.coordinatesystem = coordinatesystem
        self.computationtimeinseconds = computationtimeinseconds

    def validate(self):
        pass

    def xml_element(self):
        info = et.Element("LungCAD")
        for attr in ["Revision", "Name", "DateTimeOfExecution", "TrainingSet1", "TrainingSet2", "CoordinateSystem", "ComputationTimeInSeconds"]:
            et.SubElement(info, attr).text = str(getattr(self, attr.lower()))
        return info

    @staticmethod
    def from_xml(xml):
        revision = xml.find("./Revision").text
        name = xml.find("./Name").text
        datetimeofexecution = xml.find("./DateTimeOfExecution").text
        trainingset1 = none_str_guard(xml.find("./TrainingSet1").text)
        trainingset2 = none_str_guard(xml.find("./TrainingSet2").text)
        coordinatesystem = xml.find("./CoordinateSystem").text
        computationtimeinseconds = float(xml.find("./ComputationTimeInSeconds").text)
        return LungCad(revision, name, datetimeofexecution, trainingset1, trainingset2,
                       coordinatesystem, computationtimeinseconds)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               self.revision == other.revision and \
               self.name == other.name and \
               self.datetimeofexecution == other.datetimeofexecution and \
               self.trainingset1 == other.trainingset1 and \
               self.trainingset2 == other.trainingset2 and \
               self.coordinatesystem == other.coordinatesystem and \
               self.computationtimeinseconds == other.computationtimeinseconds

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
            self.computationtimeinseconds)


class Finding(XMLGeneratable):
    def __init__(self, id, x, y, z, probability, diameter_mm, volume_mm3):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.probability = probability
        self.diameter_mm = diameter_mm
        self.volume_mm3 = volume_mm3

    def validate(self):
        instanceofcheck(self.id, int)
        for var in [self.x, self.y, self.z, self.probability, self.diameter_mm, self.volume_mm3]:
            instanceofcheck(var, (float, int))

    def xml_element(self):
        finding = et.Element("Finding")
        for attr in ["ID", "X", "Y", "Z", "Probability", "Diameter_mm", "Volume_mm3"]:
            et.SubElement(finding, attr).text = str(getattr(self, attr.lower()))
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
        return Finding(id, x, y, z, probability, diameter_mm, volume_mm3)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               self.id == other.id and \
               self.x == other.x and \
               self.y == other.y and \
               self.z == other.z and \
               self.probability == other.probability and \
               self.diameter_mm == other.diameter_mm and \
               self.volume_mm3 == other.volume_mm3

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "id: {} x: {} y: {} z: {} probability: {} diameter_mm: {} volume_mm3: {}".format(
            self.id,
            self.x,
            self.y,
            self.z,
            self.probability,
            self.diameter_mm,
            self.volume_mm3)


class ImageInfo(XMLGeneratable):
    def __init__(self, dimensions, voxelsize, origin, orientation, patientuid, studyuid, seriesuid):
        self.dimensions = dimensions
        self.voxelsize = voxelsize
        self.origin = origin
        self.orientation = orientation
        self.patientuid = patientuid
        self.studyuid = studyuid
        self.seriesuid = seriesuid

    def validate(self):
        for var, typ in ((self.dimensions, int), (self.voxelsize, (int, float)), (self.origin, (int, float))):
            lencheck(var, 3)
            for element in var:
                instanceofcheck(element, typ)
        lencheck(self.orientation, 9)
        for element in self.orientation:
            instanceofcheck(element, (int, float))
        for var in (self.patientuid, self.studyuid, self.seriesuid):
            instanceofcheck(var, str)
            matchcheck(var, r"^[\d.]*$")

    def xml_element(self):
        info = et.Element("ImageInfo")
        for attr, subattr in (("Dimensions", "dim"), ("VoxelSize", "voxelSize"), ("Origin", "origin")):
            sub = et.SubElement(info, attr)
            for idx, element in enumerate(["X", "Y", "Z"]):
                et.SubElement(sub, "{}{}".format(subattr, element)).text = str(getattr(self, attr.lower())[idx])
        et.SubElement(info, "Orientation").text = ",".join([str(e) for e in self.orientation])
        for attr in ["PatientUID", "StudyUID", "SeriesUID"]:
            et.SubElement(info, attr).text = str(getattr(self, attr.lower()))
        return info

    @staticmethod
    def from_xml(xml):
        t = {}
        for attr, subattr, typ in (("Dimensions", "dim", int), ("VoxelSize", "voxelSize", float), ("Origin", "origin", float)):
            t[attr.lower()] = tuple([typ(xml.find("./{}/{}{}".format(attr, subattr, element)).text) for element in ["X", "Y", "Z"]])
        orientation = tuple([float(e) for e in xml.find("./Orientation").text.split(',')])
        patientuid = none_str_guard(xml.find("./PatientUID").text)
        studyuid = none_str_guard(xml.find("./StudyUID").text)
        seriesuid = xml.find("./SeriesUID").text
        patientuid = patientuid if patientuid is not None else ""
        return ImageInfo(t["dimensions"], t["voxelsize"], t["origin"], orientation, patientuid, studyuid, seriesuid)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               self.dimensions == other.dimensions and \
               self.voxelsize == other.voxelsize and \
               self.origin == other.origin and \
               self.orientation == other.orientation and \
               self.patientuid == other.patientuid and \
               self.studyuid == other.studyuid and \
               self.seriesuid == other.seriesuid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "dimensions: {} voxelsize: {} origin: {} orientation: {} patientuid: {} studyuid: {} seriesuid: {}".format(
            self.dimensions,
            self.voxelsize,
            self.origin,
            self.orientation,
            self.patientuid,
            self.studyuid,
            self.seriesuid)


class CancerInfo(XMLGeneratable):
    def __init__(self, casecancerprobability, referencenoduleids):
        self.casecancerprobability = casecancerprobability
        self.referencenoduleids = tuple(referencenoduleids)

    def validate(self):
        instanceofcheck(self.casecancerprobability, float)
        instanceofcheck(self.referencenoduleids, (list, tuple))
        for e in self.referencenoduleids:
            instanceofcheck(e, int)

    def xml_element(self):
        info = et.Element("CancerInfo")
        et.SubElement(info, "CaseCancerProbability").text = str(self.casecancerprobability)
        et.SubElement(info, "ReferenceNoduleIDs").text = ",".join([str(e) for e in self.referencenoduleids])
        return info

    @staticmethod
    def from_xml(xml):
        referencenoduleids = tuple([int(e) for e in xml.find("./ReferenceNoduleIDs").text.split(',')])
        casecancerprobability = float(xml.find("./CaseCancerProbability").text)
        return CancerInfo(casecancerprobability=casecancerprobability, referencenoduleids=referencenoduleids)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               self.referencenoduleids == other.referencenoduleids and \
               self.casecancerprobability == other.casecancerprobability

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "casecancerprobability: {} referencenoduleids: {}".format(
            self.casecancerprobability,
            self.referencenoduleids)


if __name__ == "__main__":
    finding = Finding(0, 2., 3., 4., 0.5, 20., 10.)
    lungcad = LungCad(revision= "", name="GPUCAD", datetimeofexecution="", trainingset1="", trainingset2="", coordinatesystem="World", computationtimeinseconds=33.0)
    imageinfo = ImageInfo(dimensions=(0,0,0), voxelsize=(0.,0.,0.), origin=(0.,0.,0.), orientation=(0,0,0,0,0,0,0,0,0.5), patientuid="34.2.32", studyuid="232.32.3", seriesuid="424.35")
    cancerinfo = CancerInfo(casecancerprobability=0.5, referencenoduleids=[1,2,3,4,5])

    report = LungCadReport(lungcad, imageinfo, [finding, finding], cancerinfo=cancerinfo)
    print(prettify(report.xml_element()))
