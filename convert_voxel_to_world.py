import json
import os


class ConvertVoxelToWorld(object):
    def __init__(self, prep_folder, series_uid, crop_rects_json_path,
                 output_path):
        self._conversion_parameters = {}
        self._coordinates = []
        self._preprocessing_info_file = os.path.join(prep_folder,
                                                      '{}_preprocessing_info.txt'.format(
                                                          series_uid))
        self._crop_rects_json_path = crop_rects_json_path
        self._output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.postprocessing()

    def postprocessing(self):
        self.get_conversion_parameters()
        if not all(key in self._conversion_parameters for key in
                   ['original_origin', 'extendbox_origin', 'cropped_grid_shape',
                    'resampled_spacing']):
            raise ValueError(
                "original_origin, extendbox_origin, cropped_grid_shape, resampled_spacing must be present in {}".format(
                    self._preprocessing_info_file))
        self.get_relative_voxel_coordinates()
        self.compute_cropped_voxel_coordinates()
        self.compute_world_coordinates()
        self.output_json()

    def get_conversion_parameters(self):
        if os.path.exists(self._preprocessing_info_file):
            try:
                with open(self._preprocessing_info_file, 'r') as handle:
                    lines = handle.readlines()
                    for line in lines:
                        key = line.split('=')[0]
                        values = line.split('=')[1]
                        coord = values.split(',')
                        self._conversion_parameters[key] = {
                            'x': float(coord[0]),
                            'y': float(coord[1]),
                            'z': float(coord[2].strip('\n'))}
            except IOError:
                print("Cannot read {}".format(self._preprocessing_info_file))
        else:
            print("{} does not exist".format(
                self._preprocessing_info_file))  # TODO: Should have raised an exception
        self._conversion_parameters['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                                                             'z': 1.0}

    def get_relative_voxel_coordinates(self):
        if os.path.exists(self._crop_rects_json_path):
            try:
                with open(self._crop_rects_json_path) as f:
                    cropped_rects = json.load(f)
                for items in cropped_rects.values():
                    for boundingbox in items:
                        coordinate = {"world_x": [], "world_y": [],
                                      "world_z": []}
                        for dim, min_and_max in boundingbox.iteritems():
                            coordinate['rel_voxel_{}'.format(dim)] = min_and_max
                        self._coordinates.append(coordinate)
            except IOError:
                print("Cannot read {}".format(self._crop_rects_json_path))
        else:
            print("{} does not exist".format(
                self._crop_rects_json_path))  # TODO: Should have raised an exception

    def compute_cropped_voxel_coordinates(self):
        for coord in self._coordinates:
            for dim in ['x', 'y', 'z']:
                coord['world_{}'.format(dim)] = [
                    c * self._conversion_parameters['cropped_grid_shape'][
                        '{}'.format(dim)] + \
                    self._conversion_parameters['extendbox_origin'][
                        '{}'.format(dim)] for c in
                    coord['rel_voxel_{}'.format(dim)]]

    def compute_world_coordinates(self):
        for coord in self._coordinates:
            for dim in ['x', 'y', 'z']:
                coord['world_{}'.format(dim)] = [
                    c * self._conversion_parameters['resampled_spacing'][
                        '{}'.format(dim)] + \
                    self._conversion_parameters['original_origin'][
                        '{}'.format(dim)] for c in
                    coord['world_{}'.format(dim)]]
                start_and_end = coord['world_{}'.format(dim)]
                coord['world_{}'.format(dim)] = [start_and_end[0],
                                                 sum(start_and_end) * 0.5,
                                                 start_and_end[1]]

    def output_json(self):
        with open(os.path.join(self._output_path,
                               "detected_nodules_in_world_and_voxel_coordinates.json"),
                  "w") as json_handle:
            json.dump(self._coordinates, json_handle)
