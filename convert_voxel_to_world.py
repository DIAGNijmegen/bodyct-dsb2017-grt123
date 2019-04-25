import json
import os
import copy


class ConvertVoxelToWorld(object):
    def __init__(self, prep_folder, series_uid, crop_rects_json_path,
                 output_path):
        self.__conversion_parameters = {}
        self.__boundingbox_min = {'x': [], 'y': [], 'z': []}
        self.__boundingbox_max = {'x': [], 'y': [], 'z': []}
        self.__boundingbox_centers = {'x': [], 'y': [], 'z': []}
        self.__original_boundingbox_min = {}
        self.__original_boundingbox_max = {}
        self.__preprocessing_info_file = os.path.join(prep_folder,
                                                      '{}_preprocessing_info.txt'.format(
                                                          series_uid))
        self.__crop_rects_json_path = crop_rects_json_path
        self.__output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def postprocessing(self):
        self.get_conversion_parameters()
        if not all(key in self.__conversion_parameters for key in
                   ['original_origin', 'extendbox_origin', 'cropped_grid_shape',
                    'resampled_spacing']):
            raise ValueError(
                "original_origin, extendbox_origin, cropped_grid_shape, resampled_spacing must be present in {}".format(
                    self.__preprocessing_info_file))
        self.get_relative_voxel_coordinates()
        self.compute_cropped_voxel_coordinates()
        self.compute_world_coordiantes()
        self.output_json()

    def get_conversion_parameters(self):
        if os.path.exists(self.__preprocessing_info_file):
            try:
                with open(self.__preprocessing_info_file, 'r') as handle:
                    lines = handle.readlines()
                    for line in lines:
                        key = line.split('=')[0]
                        values = line.split('=')[1]
                        coord = values.split(',')
                        self.__conversion_parameters[key] = {
                            'x': float(coord[0]),
                            'y': float(coord[1]),
                            'z': float(coord[2].strip('\n'))}
            except IOError:
                print("Cannot read {}".format(self.__preprocessing_info_file))
        else:
            print("{} does not exist".format(
                self.__preprocessing_info_file))  # TODO: Should have raised an exception
        self.__conversion_parameters['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                                                             'z': 1.0}

    def get_relative_voxel_coordinates(self):
        if os.path.exists(self.__crop_rects_json_path):
            try:
                with open(self.__crop_rects_json_path) as f:
                    cropped_rects = json.load(f)
                for key, items in cropped_rects.iteritems():
                    for boundingbox in items:
                        for dim, min_and_max in boundingbox.iteritems():
                            self.__boundingbox_min[dim].append(
                                float(min_and_max[0]))
                            self.__boundingbox_max[dim].append(
                                float(min_and_max[1]))
            except IOError:
                print("Cannot read {}".format(self.__crop_rects_json_path))
        else:
            print("{} does not exist".format(
                self.__crop_rects_json_path))  # TODO: Should have raised an exception
        self.__original_boundingbox_min = copy.deepcopy(self.__boundingbox_min)
        self.__original_boundingbox_max = copy.deepcopy(self.__boundingbox_max)
        print("Original bounding box")
        print(self.__original_boundingbox_min)
        print(self.__original_boundingbox_max)

    def compute_cropped_voxel_coordinates(self):
        for dim, value in self.__conversion_parameters[
            'extendbox_origin'].iteritems():
            for index, min_coord in enumerate(
                    self.__boundingbox_min[dim]):
                self.__boundingbox_min[dim][index] = min_coord * \
                                                     self.__conversion_parameters[
                                                         'cropped_grid_shape'][
                                                         dim] + value
        for dim, value in self.__conversion_parameters[
            'extendbox_origin'].iteritems():
            for index, max_coord in enumerate(
                    self.__boundingbox_max[dim]):
                self.__boundingbox_max[dim][index] = max_coord * \
                                                     self.__conversion_parameters[
                                                         'cropped_grid_shape'][
                                                         dim] + value

    def compute_world_coordiantes(self):
        for dim, value in self.__conversion_parameters[
            'original_origin'].iteritems():
            for index, min_coord in enumerate(
                    self.__boundingbox_min[dim]):
                self.__boundingbox_min[dim][index] = min_coord * \
                                                     self.__conversion_parameters[
                                                         'resampled_spacing'][
                                                         dim] + value

        for dim, value in self.__conversion_parameters[
            'original_origin'].iteritems():
            for index, max_coord in enumerate(
                    self.__boundingbox_max[dim]):
                self.__boundingbox_max[dim][index] = max_coord * \
                                                     self.__conversion_parameters[
                                                         'resampled_spacing'][
                                                         dim] + value
        for dim, nodule_min_bounding_box in self.__boundingbox_min.iteritems():
            for index, min_coord in enumerate(nodule_min_bounding_box):
                self.__boundingbox_centers[dim].append(
                    min_coord + (self.__boundingbox_max[dim][
                                     index] - min_coord) * 0.5)

    def output_json(self):
        # Compute input to the listString field of MeVisLab
        counter = len(self.__boundingbox_centers['x'])
        output_string = '['
        for coord_x, coord_y, coord_z in zip(self.__boundingbox_centers['x'],
                                             self.__boundingbox_centers['y'],
                                             self.__boundingbox_centers['z']):
            output_string += '({} {} {}) #{}, '.format(coord_x, coord_y,
                                                       coord_z,
                                                       counter)
            counter = counter - 1
        output_string = output_string[:-2]
        output_string += ']'
        print(output_string)

        number_of_nodules = len(self.__boundingbox_centers['x'])
        output_json_dict = []
        for i in range(number_of_nodules):
            output_json_dict_per_nodule = {"world_x": [], "world_y": [],
                                           "world_z": [], "rel_voxel_x": [],
                                           "rel_voxel_y": [], "rel_voxel_z": []}
            output_json_dict_per_nodule["world_x"].append(
                self.__boundingbox_min['x'][i])
            output_json_dict_per_nodule["world_x"].append(
                self.__boundingbox_centers['x'][i])
            output_json_dict_per_nodule["world_x"].append(
                self.__boundingbox_max['x'][i])
            output_json_dict_per_nodule["world_y"].append(
                self.__boundingbox_min['y'][i])
            output_json_dict_per_nodule["world_y"].append(
                self.__boundingbox_centers['y'][i])
            output_json_dict_per_nodule["world_y"].append(
                self.__boundingbox_max['y'][i])
            output_json_dict_per_nodule["world_z"].append(
                self.__boundingbox_min['z'][i])
            output_json_dict_per_nodule["world_z"].append(
                self.__boundingbox_centers['z'][i])
            output_json_dict_per_nodule["world_z"].append(
                self.__boundingbox_max['z'][i])
            output_json_dict_per_nodule["rel_voxel_x"].append(
                self.__original_boundingbox_min['x'][i])
            output_json_dict_per_nodule["rel_voxel_x"].append(
                self.__original_boundingbox_max['x'][i])
            output_json_dict_per_nodule["rel_voxel_y"].append(
                self.__original_boundingbox_min['y'][i])
            output_json_dict_per_nodule["rel_voxel_y"].append(
                self.__original_boundingbox_max['y'][i])
            output_json_dict_per_nodule["rel_voxel_z"].append(
                self.__original_boundingbox_min['z'][i])
            output_json_dict_per_nodule["rel_voxel_z"].append(
                self.__original_boundingbox_max['z'][i])

            output_json_dict.append(output_json_dict_per_nodule)
        with open(os.path.join(self.__output_path,
                               "detected_nodules_in_world_and_voxel_coordinates.json"),
                  "w") as json_handle:
            json.dump(output_json_dict, json_handle)


def convert_voxel_to_world(prep_folder, series_uid, crop_rects_json_path,
                           output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    input_dict = {}
    preprocessing_info_file_name = os.path.join(prep_folder,
                                                '{}_preprocessing_info.txt'.format(
                                                    series_uid))
    if os.path.exists(preprocessing_info_file_name):
        try:
            with open(preprocessing_info_file_name, 'r') as handle:
                lines = handle.readlines()
                for line in lines:
                    key = line.split('=')[0]
                    values = line.split('=')[1]
                    coord = values.split(',')
                    input_dict[key] = {'x': float(coord[0]),
                                       'y': float(coord[1]),
                                       'z': float(coord[2].strip('\n'))}
        except IOError:
            print("Cannot read {}".format(preprocessing_info_file_name))
    else:
        print("{} does not exist".format(preprocessing_info_file_name))
        return
    input_dict['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                                       'z': 1.0}
    boundingbox_min = {'x': [], 'y': [], 'z': []}
    boundingbox_max = {'x': [], 'y': [], 'z': []}
    if os.path.exists(crop_rects_json_path):
        try:
            with open(crop_rects_json_path) as f:
                cropped_rects = json.load(f)
            for key, items in cropped_rects.iteritems():
                for boundingbox in items:
                    for dim, min_and_max in boundingbox.iteritems():
                        boundingbox_min[dim].append(float(min_and_max[0]))
                        boundingbox_max[dim].append(float(min_and_max[1]))
        except IOError:
            print("Cannot read {}".format(crop_rects_json_path))

    print("Original bounding box")
    original_boundingbox_min = copy.deepcopy(boundingbox_min)
    original_boundingbox_max = copy.deepcopy(boundingbox_max)
    print(original_boundingbox_min)
    print(original_boundingbox_max)
    for dim, value in input_dict['extendbox_origin'].iteritems():
        for index, min_coord in enumerate(
                boundingbox_min[dim]):
            boundingbox_min[dim][index] = min_coord * \
                                          input_dict['cropped_grid_shape'][
                                              dim] + value
    for dim, value in input_dict['extendbox_origin'].iteritems():
        for index, max_coord in enumerate(
                boundingbox_max[dim]):
            boundingbox_max[dim][index] = max_coord * \
                                          input_dict['cropped_grid_shape'][
                                              dim] + value

    print(boundingbox_min)
    print(boundingbox_max)

    for dim, value in input_dict['original_origin'].iteritems():
        for index, min_coord in enumerate(
                boundingbox_min[dim]):
            boundingbox_min[dim][index] = min_coord * \
                                          input_dict['resampled_spacing'][
                                              dim] + value

    for dim, value in input_dict['original_origin'].iteritems():
        for index, max_coord in enumerate(
                boundingbox_max[dim]):
            boundingbox_max[dim][index] = max_coord * \
                                          input_dict['resampled_spacing'][
                                              dim] + value

    print(boundingbox_min)
    print(boundingbox_max)

    nodule_centers = {'x': [], 'y': [], 'z': []}

    for dim, nodule_min_bounding_box in boundingbox_min.iteritems():
        for index, min_coord in enumerate(nodule_min_bounding_box):
            nodule_centers[dim].append(
                min_coord + (boundingbox_max[dim][index] - min_coord) * 0.5)
    print("Centers are ")
    print(nodule_centers)
    counter = len(nodule_centers['x'])
    output_string = '['
    for coord_x, coord_y, coord_z in zip(nodule_centers['x'],
                                         nodule_centers['y'],
                                         nodule_centers['z']):
        output_string += '({} {} {}) #{}, '.format(coord_x, coord_y, coord_z,
                                                   counter)
        counter = counter - 1
    output_string = output_string[:-2]
    output_string += ']'
    print(output_string)
    # with open(os.path.join(output_path, "converted_voxel_to_world.txt"),
    #           "w+") as output:
    #     output.write(output_string)

    # TODO: Cleanup
    # Output the new json format
    number_of_nodules = len(nodule_centers['x'])
    output_json_dict = []
    for i in range(0, number_of_nodules):
        output_json_dict_per_nodule = {"world_x": [], "world_y": [],
                                       "world_z": [], "rel_voxel_x": [],
                                       "rel_voxel_y": [], "rel_voxel_z": []}
        output_json_dict_per_nodule["world_x"].append(boundingbox_min['x'][i])
        output_json_dict_per_nodule["world_x"].append(nodule_centers['x'][i])
        output_json_dict_per_nodule["world_x"].append(boundingbox_max['x'][i])
        output_json_dict_per_nodule["world_y"].append(boundingbox_min['y'][i])
        output_json_dict_per_nodule["world_y"].append(nodule_centers['y'][i])
        output_json_dict_per_nodule["world_y"].append(boundingbox_max['y'][i])
        output_json_dict_per_nodule["world_z"].append(boundingbox_min['z'][i])
        output_json_dict_per_nodule["world_z"].append(nodule_centers['z'][i])
        output_json_dict_per_nodule["world_z"].append(boundingbox_max['z'][i])
        output_json_dict_per_nodule["rel_voxel_x"].append(
            original_boundingbox_min['x'][i])
        output_json_dict_per_nodule["rel_voxel_x"].append(
            original_boundingbox_max['x'][i])
        output_json_dict_per_nodule["rel_voxel_y"].append(
            original_boundingbox_min['y'][i])
        output_json_dict_per_nodule["rel_voxel_y"].append(
            original_boundingbox_max['y'][i])
        output_json_dict_per_nodule["rel_voxel_z"].append(
            original_boundingbox_min['z'][i])
        output_json_dict_per_nodule["rel_voxel_z"].append(
            original_boundingbox_max['z'][i])

        output_json_dict.append(output_json_dict_per_nodule)
    with open(os.path.join(output_path,
                           "detected_nodules_in_world_and_voxel_coordinates.json"),
              "w") as json_handle:
        json.dump(output_json_dict, json_handle)
