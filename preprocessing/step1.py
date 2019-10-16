from __future__ import print_function

import threading
import numpy as np
import os
import sys
import scipy.ndimage
import SimpleITK as sitk

from skimage import measure

try:
    import image_loader as diag_image_loader
except ImportError:
    print(
        "(diag-)image_loader not found, loading dicom will not be possible",
        file=sys.stderr)
    diag_image_loader = None


class ParallelCaller(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.__result = None
        self.thread = threading.Thread(target=self)
        self.thread.start()

    def __call__(self, *args, **kwargs):
        self.__result = self.cmd(*args, **kwargs)

    @property
    def result(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        return self.__result


def load_dicom_scan(data_path, prep_folder, name):
    if diag_image_loader is None:
        return None
    case_path = os.path.join(data_path, name)
    print("case_path={}, data_path={}, name={}".format(case_path, data_path, name))
    image, transform, origin, spacing = diag_image_loader.load_dicom_image(
        [os.path.join(case_path, fn) for fn in os.listdir(case_path)])
    shape = [e for e in reversed(image.shape)]
    preprocessing_info_file_name = os.path.join(
        prep_folder,
        '{}_preprocessing_info.txt'.format(name))

    write_image_info_to_file(preprocessing_info_file_name, transform, origin, spacing, shape)

    return np.array(image, dtype=np.int16), np.array(spacing, dtype=np.float32)


def load_itk_image(path, prep_folder):
    sitk_image = sitk.ReadImage(path)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    transform = np.array(sitk_image.GetDirection()).reshape((3, 3))
    shape = sitk_image.GetSize()
    pixel_data = sitk.GetArrayFromImage(sitk_image)
    preprocessing_info_file_name = os.path.join(
        prep_folder,
        '{}_preprocessing_info.txt'.format(os.path.basename(os.path.normpath(path))))

    write_image_info_to_file(preprocessing_info_file_name, transform, origin, spacing, shape)

    return np.array(pixel_data, dtype=np.int16), np.array(
        sitk_image.GetSpacing(), dtype=np.float32)


def write_image_info_to_file(fname, transform, origin, spacing, shape):
    with open(fname, 'w') as handle:
        handle.write(
            'rotation_matrix_x={},{},{}\n'.format(float(transform[2][2]),
                                                  float(transform[2][1]),
                                                  float(transform[2][0])))
        handle.write(
            'rotation_matrix_y={},{},{}\n'.format(float(transform[1][2]),
                                                  float(transform[1][1]),
                                                  float(transform[1][0])))
        handle.write(
            'rotation_matrix_z={},{},{}\n'.format(float(transform[0][2]),
                                                  float(transform[0][1]),
                                                  float(transform[0][0])))
        handle.write(
            'original_origin={},{},{}\n'.format(float(origin[2]),
                                                float(origin[1]),
                                                float(origin[0])))
        handle.write(
            'original_spacing={},{},{}\n'.format(float(spacing[2]),
                                                 float(spacing[1]),
                                                 float(spacing[0])))
        handle.write(
            'original_shape={},{},{}\n'.format(int(shape[2]),
                                               int(shape[1]),
                                               int(shape[0])))


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30,
                       eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5,
                            image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(
                np.multiply(image[i].astype('float32'), nan_mask), sigma,
                truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(
                image[i].astype('float32'), sigma, truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[
                2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2],
                       area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set(
        [label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
         label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1],
         label[-1 - cut_num, -1, 0], label[-1 - cut_num, -1, -1], \
         label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid],
         label[-1 - cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[
            0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5,
                         label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5,
                         label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(
                single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if
                       slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set(
        [label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
         label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0],
         label[-1, -1, -1]])
    bw = ~np.isin(label, list(bg_label))

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2],
                                                   bb[1]:bb[3]] | properties[
                                                       j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        def run_hole_filler(i):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[
                                                          bb[0]:bb[2], bb[1]:bb[
                    3]] | prop.filled_image
            bw[i] = current_slice

        threads = [threading.Thread(target=run_hole_filler, args=(i,)) for i in
                   range(bw.shape[0])]
        for t in threads: t.start()
        for t in threads: t.join()

        return bw

    def sequential_eroder(bw):
        found_flag = False
        iter_count = 0
        bw0 = np.copy(bw)
        bw1 = None
        bw2 = None

        while not found_flag and iter_count < max_iter:
            label = measure.label(bw, connectivity=2)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            if len(properties) > 1 and float(properties[0].area) / float(
                    properties[1].area) < max_ratio:
                found_flag = True
                bw1 = label == properties[0].label
                bw2 = label == properties[1].label
            else:
                bw = scipy.ndimage.binary_erosion(bw)
                iter_count = iter_count + 1

        return found_flag, bw, bw0, bw1, bw2

    found_flag, bw, bw0, bw1, bw2 = sequential_eroder(bw)

    if found_flag:
        c1 = ParallelCaller(
            lambda: scipy.ndimage.morphology.distance_transform_edt(
                bw1 == False, sampling=spacing))
        c2 = ParallelCaller(
            lambda: scipy.ndimage.morphology.distance_transform_edt(
                bw2 == False, sampling=spacing))
        d1 = c1.result
        d2 = c2.result

        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


import time


def step1_python(data_path, prep_folder, name):
    st = time.time()
    case_path = os.path.join(data_path, name)
    print("  Loading", case_path)
    if os.path.isdir(case_path):
        scan_data = load_dicom_scan(data_path, prep_folder, name)
        if scan_data is None:
            return None
        case_pixels, spacing = scan_data
    elif os.path.splitext(case_path)[-1].lower() in ('.mha', '.mhd'):
        case_pixels, spacing = load_itk_image(case_path, prep_folder)
    else:
        raise ValueError("Unknown file type: " + case_path)

    print("binarize...", time.time() - st)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    print("slicing...", time.time() - st)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num,
                                      vol_limit=[0.68, 7.5])
        cut_num = cut_num + cut_step

    print("fill_hole...", time.time() - st)
    bw = fill_hole(bw)
    print("two_lung_only...", time.time() - st)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    print("end step1", time.time() - st)
    return case_pixels, bw1, bw2, spacing


if __name__ == '__main__':
    INPUT_FOLDER = '/work/DataBowl3/stage1/stage1/'
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    case_pixels, m1, m2, spacing = step1_python(
        os.path.join(INPUT_FOLDER, patients[25]), os.path.join(INPUT_FOLDER, patients[25]))
    plt.imshow(m1[60])
    plt.figure()
    plt.imshow(m2[60])
#     first_patient = load_scan(INPUT_FOLDER + patients[25])
#     first_patient_pixels, spacing = get_pixels_hu(first_patient)
#     plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()

#     # Show some slice in the middle
#     h = 80
#     plt.imshow(first_patient_pixels[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = binarize_per_slice(first_patient_pixels, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     flag = 0
#     cut_num = 0
#     while flag == 0:
#         bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num)
#         cut_num = cut_num + 1
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = fill_hole(bw)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw1, bw2, bw = two_lung_only(bw, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
