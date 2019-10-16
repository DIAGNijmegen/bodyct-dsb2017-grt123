from test_predict import ensure_testdata_unpacked, get_config
from pathlib2 import Path
import numpy as np
import SimpleITK as sitk
import json
import main
import pytest


def get_info(fname):
    with open(fname, "r") as f:
        lines = [line.strip().split("=") for line in f.readlines()]
    d = {key: np.array([float(e) for e in reversed(values.split(","))]) for key, values in lines}
    rotmatrix = np.array([[ee for ee in reversed(d["rotation_matrix_{}".format(e)])] for e in ["x", "y", "z"]])
    d["rotation_matrix"] = rotmatrix
    return dict(
        rotation_matrix = rotmatrix,
        origin = d["original_origin"],
        spacing = d["original_spacing"],
        shape = d["original_shape"].astype(np.int),
        crop_shape = d["cropped_grid_shape"].astype(np.int),
        crop_origin = d["extendbox_origin"].astype(np.int)
    )


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


def test_coord_relative_to_world():
    write_screenshots = False
    ensure_testdata_unpacked()
    res_dir = Path(__file__).parent / "resources"
    coords_file = res_dir / "debug.json"
    mhd_file = res_dir / "inputs" / "lidc.mhd"
    info_file = res_dir / "lidc.mhd_preprocessing_info.txt"
    # coords_file = res_dir / "temp_results_generated.json"
    # mhd_file = res_dir / "1.2.276.0.7230010.3.1.3.358940797.10148.1541496242.15553.mhd"
    # info_file = res_dir / "temp_preprocessing_info.txt"

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
    cfg = get_config(tmp_path, test_data_dir)
    res_dir = Path(__file__).parent / "resources"
    mhd_file = res_dir / "inputs" / "lidc.mhd"
    with pytest.raises(IOError):
        main.main(
            skip_detect=True,
            skip_preprocessing=False,
            **cfg
        )
    image = sitk.ReadImage(str(mhd_file))

    for f in ["lidc-dcm", "lidc.mhd", "lidc.mha"]:
        info_file = tmp_path / "prep" / (f + "_preprocessing_info.txt")
        assert info_file.exists()
        info = get_info(str(info_file))

        assert np.allclose(info["rotation_matrix"], np.array(image.GetDirection()).reshape((3,3)))
        assert np.allclose(info["origin"], np.array(image.GetOrigin()))
        assert np.allclose(info["spacing"], np.array(image.GetSpacing()))
        assert np.allclose(info["shape"], np.array(image.GetSize()))
