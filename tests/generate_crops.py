import SimpleITK as sitk
from pathlib2 import Path
import numpy as np


def get_info(fname):
    with open(fname, "r") as f:
        lines = [line.strip().split("=") for line in f.readlines()]
    return {key: np.array([float(e) for e in reversed(values.split(","))]) for key, values in lines}


def generate_crops():
    resdir = Path(__file__).parent / "resources" / "inputs2"
    for i in [0, 3, 43]:
        print(i)
        inputfile = str(resdir / "nodules_{}.mhd".format(i))

        infofile = str(inputfile).replace(".mhd", ".txt")
        outfile = str(inputfile).replace(".mhd", "_res.mhd")

        infos = get_info(infofile)
        corigin, cshape = infos["extendbox_origin"].astype(np.int), infos["cropped_grid_shape"].astype(np.int)

        # read image
        image = sitk.ReadImage(inputfile)
        data = sitk.GetArrayFromImage(image)

        # crop image
        c = -3024
        data[:corigin[0], :, :] = c
        data[:, :corigin[1], :] = c
        data[:, :, :corigin[2]] = c
        data[corigin[0]+cshape[0]:, :, :] = c
        data[:, corigin[1]+cshape[1]:, :] = c
        data[:, :, corigin[2]+cshape[2]:] = c

        # write to output
        imageout = sitk.GetImageFromArray(data, isVector=False)
        imageout.SetSpacing(image.GetSpacing())
        imageout.SetOrigin(image.GetOrigin())
        imageout.SetDirection(image.GetDirection())
        sitk.WriteImage(imageout, outfile, True)


def generate_test_image():
    outfile = Path(__file__).parent / "resources" / "test.mhd"
    data = np.zeros((160, 512, 512), dtype=np.int16)
    size = 5
    # z, y, x
    for pos in [
        [0, 0, 0],
        [160, 512, 512],
        [80, 256, 256],
        [50, 60, 70],
        [150, 60, 70]
    ]:
        minpos = np.maximum(np.array(pos) - size, [0, 0, 0])
        maxpos = np.minimum(np.array(pos) + size, data.shape)
        data[minpos[0]:maxpos[0], minpos[1]:maxpos[1], minpos[2]:maxpos[2]] = 100

    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, str(outfile), True)


def generate_mha_for_mhd(mhd_file):
    infilea = Path(mhd_file)
    outfile = mhd_file.parent / mhd_file.name.replace(".mhd", ".mha")
    infileb = mhd_file.parent / mhd_file.name.replace(".mhd", ".zraw")
    with open(str(outfile), "wb") as f:
        with open(str(infilea), "r") as header:
            hdr = header.read().replace(str(infileb.name), 'LOCAL')
        f.write(hdr)
        with open(str(infileb), "rb") as body:
            f.write(body.read())


def generate_4d_test_image():
    outfile = Path(__file__).parent / "resources" / "image10x11x12x13.mhd"
    data = np.zeros((10, 11, 12, 13), dtype=np.uint8)
    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, str(outfile), True)


if __name__ == "__main__":
    generate_mha_for_mhd(Path(__file__).absolute().parent.parent.parent / "grand-challenge.org" / "app" / "tests" / "cases_tests" / "resources" / "image10x11x12x13.mhd")