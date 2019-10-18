from test_predict import ensure_testdata_unpacked
from pathlib2 import Path
import numpy as np
import SimpleITK as sitk
import json
import pytest
import shutil
from convert_voxel_to_world import ConvertVoxelToWorld
from preprocessing.step1 import load_itk_image, load_image
import os


def get_info(fname):
    with open(fname, "r") as f:
        lines = [line.strip().split("=") for line in f.readlines()]
    d = {key: np.array([float(e) for e in reversed(values.split(","))]) for key, values in lines}
    rotmatrix = np.array([[ee for ee in d["rotation_matrix_{}".format(e)]] for e in ["z", "y", "x"]])
    d["rotation_matrix"] = rotmatrix
    info = dict(
        rotation_matrix = rotmatrix,
        origin = d["original_origin"],
        spacing = d["original_spacing"],
        shape = d["original_shape"].astype(np.int)
    )
    if "cropped_grid_shape" in d:
        info["crop_shape"] = d["cropped_grid_shape"].astype(np.int)
    if "extendbox_origin" in d:
        info["crop_origin"] = d["extendbox_origin"].astype(np.int)
    return info


def read_json_coordinates(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    r = dict()
    for k, v in data.items():
        nodlist = []
        for e in v:
            nodlist.append(dict(
                rel_voxel_min=np.array([e['rel_voxel_{}'.format(i)][0] for i in ['x', 'y', 'z']]),
                rel_voxel_max=np.array([e['rel_voxel_{}'.format(i)][1] for i in ['x', 'y', 'z']]),
                world_voxel_min=np.array([e['world_{}'.format(i)][0] for i in ['x', 'y', 'z']]),
                world_voxel_mean=np.array([e['world_{}'.format(i)][1] for i in ['x', 'y', 'z']]),
                world_voxel_max=np.array([e['world_{}'.format(i)][2] for i in ['x', 'y', 'z']]),
            ))
        r[k] = nodlist

    return r


def test_match_metadata_order_xyz_with_mhd_header(tmp_path):
    res_dir = Path(__file__).parent / "resources"
    prep_dir = tmp_path / "prep"
    os.makedirs(str(prep_dir))
    mhd_file = res_dir / "inputs" / "lidc.mhd"
    load_itk_image(str(mhd_file), str(prep_dir))
    info_file = prep_dir / (mhd_file.name + "_preprocessing_info.txt")
    assert info_file.exists()
    with open(str(info_file), "r") as f:
        lines = [line.strip().split("=") for line in f.readlines()]
    d = {key: np.array([float(e) for e in values.split(",")]) for key, values in lines}
    rotmatrix = np.array([[ee for ee in d["rotation_matrix_{}".format(e)]] for e in ["x", "y", "z"]]).flatten()
    image = sitk.ReadImage(str(mhd_file))
    assert np.allclose(image.GetSpacing(), d["original_spacing"])
    assert np.allclose(image.GetOrigin(), d["original_origin"])
    assert np.allclose(image.GetSize(), d["original_shape"])
    assert np.allclose(image.GetDirection(), rotmatrix)


def test_coord_relative_to_world():
    write_screenshots = False
    ensure_testdata_unpacked()
    res_dir = Path(__file__).parent / "resources"
    coords_file = res_dir / "debug.json"
    mhd_file = res_dir / "inputs" / "lidc.mhd"
    info_file = res_dir / "lidc.mhd_preprocessing_info.txt"

    info = get_info(str(info_file))

    spacing = info["spacing"]
    crop_shape = info["crop_shape"]
    crop_origin = info["crop_origin"]

    voxel_crop_shape = crop_shape / spacing
    voxel_crop_origin = crop_origin / spacing

    image = sitk.ReadImage(str(mhd_file))
    data = None
    coordinfos = read_json_coordinates(str(coords_file))
    for i, coordinfo in enumerate(coordinfos[str(mhd_file.name)]):
        print(i)
        rel_coord_a = np.array( [e for e in reversed(coordinfo["rel_voxel_min"])])  # z, y, x
        rel_coord_b = np.array( [e for e in reversed(coordinfo["rel_voxel_max"])])  # z, y, x
        print(rel_coord_a)
        print(rel_coord_b)

        rel_coord = (rel_coord_b - rel_coord_a) * 0.5 + rel_coord_a  # z, y, x
        assert np.allclose(rel_coord, (rel_coord_b + rel_coord_a) * 0.5)
        coord = voxel_crop_origin + voxel_crop_shape * rel_coord  # z, y, x

        wcoord = [a for a in reversed(image.TransformContinuousIndexToPhysicalPoint([e for e in reversed(coord.tolist())]))]  # z, y, x
        wcoord_from_info = [float(e) for e in reversed(coordinfo["world_voxel_mean"])]  # z, y, x
        print(coord)  # z, y, x
        print(wcoord)  # z, y, x
        print(wcoord_from_info)  # z, y, x
        assert np.allclose(wcoord, wcoord_from_info)

        if write_screenshots:
            if data is None:
                data = sitk.GetArrayFromImage(image)
            screensdir = res_dir / "tempscreens"

            import matplotlib.pyplot as plt
            slc = data[int(coord[0]), :, :]  # z, y, x
            plt.figure()
            plt.imshow(slc)
            plt.scatter(int(coord[2]), int(coord[1]), s=3, c='red', marker='o')  # x, y
            plt.savefig(str(screensdir / "coord{}.png".format(i)), bbox_inches="tight")
            plt.close()


def test_correct_imageinfos_are_created(tmp_path):
    test_data_dir = ensure_testdata_unpacked()
    prep_dir = tmp_path / "prep"
    os.makedirs(str(prep_dir))
    res_dir = Path(__file__).parent / "resources"
    mhd_file = res_dir / "inputs" / "lidc.mhd"
    image = sitk.ReadImage(str(mhd_file))
    for f in ["lidc-dcm", "lidc.mhd", "lidc.mha"]:
        load_image(str(test_data_dir), str(prep_dir), f)
        info_file = prep_dir / (f + "_preprocessing_info.txt")
        assert info_file.exists()
        info = get_info(str(info_file))
        assert np.allclose(info["rotation_matrix"], np.array([e for e in reversed(image.GetDirection())]).reshape((3,3)))
        assert np.allclose(info["origin"], np.array([e for e in reversed(image.GetOrigin())]))
        assert np.allclose(info["spacing"], np.array([e for e in reversed(image.GetSpacing())]))
        assert np.allclose(info["shape"], np.array([e for e in reversed(image.GetSize())]))


# x y z - ordering (similar to MHD headers)
@pytest.mark.parametrize("transform_matrix", [
    np.eye(3, 3),
    #np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
])
@pytest.mark.parametrize("voxel_crop_origin", [
    np.array([0, 0, 0]),
    np.array([10, 20, 30]),
])
@pytest.mark.parametrize("voxel_crop_shape", [
    np.array([512, 512, 160]),
    np.array([256, 256, 80]),
    np.array([45, 50, 62]),
])
@pytest.mark.parametrize("offset", [
    np.array([0, 0, 0]),
    np.array([-379, -210, -228.80000305175781]),
    np.array([-148.11089999999999, -159.04839999999999, 1576])
])
@pytest.mark.parametrize("spacing", [
    np.array([1, 1, 1]),
    np.array([2.5, 0.8203120231628418, 0.8203120231628418]),
    np.array([0.65299999713897705, 0.65299999713897705, 0.5])
])
def test_voxel_to_world_conversion(tmp_path, transform_matrix, offset, spacing, voxel_crop_origin, voxel_crop_shape):
    resdir = Path(__file__).parent / "resources"
    testfile = resdir / "test.mhd"
    testdatafile = resdir / "test.zraw"
    tmptestfile = tmp_path / testfile.name
    prepdir = tmp_path / "prep"
    os.makedirs(str(prepdir))
    shutil.copy(str(testfile), str(tmptestfile))
    shutil.copy(str(testdatafile), str(tmp_path / testdatafile.name))
    with open(str(tmptestfile), "r") as f:
        header = f.read()
    header = header.replace("TransformMatrix = 1 0 0 0 1 0 0 0 1", "TransformMatrix = {}".format(
        ' '.join(
        [str(e) for e in transform_matrix.flatten().tolist()]
        )))
    header = header.replace("Offset = 0 0 0", "Offset = {}".format(
        ' '.join(
        [str(e) for e in offset.tolist()]
        )))
    header = header.replace("ElementSpacing = 1 1 1", "ElementSpacing = {}".format(
        ' '.join(
        [str(e) for e in spacing.tolist()]
        )))
    print(header)
    with open(str(tmptestfile), "w") as f:
        f.write(header)

    image = sitk.ReadImage(str(tmptestfile))
    imageshape = np.array(image.GetSize())  # x, y, z

    # create image info and inject bounding box information
    load_itk_image(str(tmp_path / "test.mhd"), str(prepdir))
    with open(str(prepdir / "test.mhd_preprocessing_info.txt"), "a+") as f:
        f.write("extendbox_origin={}\ncropped_grid_shape={}\n".format(
            ','.join([str(e) for e in ((voxel_crop_origin * spacing).tolist())]),  # x, y, z
            ','.join([str(e) for e in ((voxel_crop_shape * spacing).tolist())])  # x, y, z
        ))
    with open(str(prepdir / "test.mhd_preprocessing_info.txt"), "r") as f:
        print(f.read())

    wcoords = []
    vcoords = []
    rectlist = []
    # z, y, x
    for vcoord in [
        [0, 0, 0],
        [160, 512, 512],
        [80, 256, 256],
        [50, 60, 70],
        [150, 60, 70]
    ]:
        vcoord = [e for e in reversed(vcoord)]
        wcoord = image.TransformContinuousIndexToPhysicalPoint(vcoord)  # x, y, z

        # compute rects x, y, z order...
        rectlist.append({
           key: ((vcoord[i]-10-voxel_crop_origin[i]) / float(voxel_crop_shape[i]),
                 (vcoord[i]+10-voxel_crop_origin[i]) / float(voxel_crop_shape[i])) for i, key in enumerate(['x', 'y', 'z'])
        })
        wcoords.append(wcoord)
        vcoords.append(vcoord)

    jsonfile = tmp_path / "test.json"
    rects = {"test.mhd": rectlist}
    print("")
    ConvertVoxelToWorld(str(prepdir), cropped_rects=rects, output_file=str(jsonfile))
    for i, res in enumerate(read_json_coordinates(str(jsonfile))["test.mhd"]):
        print("{:40} {:40} {:40}".format(str(vcoords[i]), str(wcoords[i]), str(res["world_voxel_mean"])))
        assert np.allclose(wcoords[i], res["world_voxel_mean"])
