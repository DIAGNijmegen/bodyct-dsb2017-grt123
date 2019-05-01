import json
import os

world = 'world_{}'
voxel = 'rel_voxel_{}'
dimensions = ['x', 'y', 'z']


class ConvertVoxelToWorld(object):
    def __init__(self, preprocessing_info_dir, cropped_rects,
                 output_path):
        self._conversion_parameters = {}
        self._coordinates = {}
        self._preprocessing_info_dir = preprocessing_info_dir
        self._dict_of_list_of_cropped_rects = cropped_rects
        self._output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.postprocessing()

    def postprocessing(self):
        self.get_relative_voxel_coordinates()
        self.get_conversion_parameters()
        for series_uid, conversion_parameter in self._conversion_parameters.iteritems():
            print(
                "series_uid = {}, conversion_parameter = {}".format(series_uid,
                                                                    conversion_parameter))
            if not all(key in conversion_parameter.keys() for key in
                       ['original_origin', 'extendbox_origin',
                        'cropped_grid_shape',
                        'resampled_spacing']):
                raise ValueError(
                    'original_origin, extendbox_origin, cropped_grid_shape, resampled_spacing must be present in {}'.format(
                        os.path.join(self._preprocessing_info_dir,
                                     '{}_preprocessing_info.txt'.format(
                                         series_uid))))
        self.compute_cropped_voxel_coordinates()
        self.compute_world_coordinates()
        self.output_json()

    def get_relative_voxel_coordinates(self):
        for series_uid, cropped_rects in self._dict_of_list_of_cropped_rects.iteritems():
            self._coordinates[series_uid] = [
                {voxel.format(dim): min_and_max for dim, min_and_max
                 in boundingbox.iteritems()} for boundingbox in
                cropped_rects]

    def get_conversion_parameters(self):
        for series_uid in self._coordinates.keys():
            preprocessing_info_file = os.path.join(self._preprocessing_info_dir,
                                                   '{}_preprocessing_info.txt'.format(
                                                       series_uid))
            conversion_parameter = {}
            if os.path.exists(preprocessing_info_file):
                try:
                    with open(preprocessing_info_file, 'r') as handle:
                        lines = handle.readlines()
                        for line in lines:
                            key = line.split('=')[0]
                            values = line.split('=')[1]
                            coord = values.split(',')
                            conversion_parameter[key] = {
                                'x': float(coord[0]),
                                'y': float(coord[1]),
                                'z': float(coord[2].strip('\n'))}
                    conversion_parameter['resampled_spacing'] = {
                        'x': 1.0,
                        'y': 1.0,
                        'z': 1.0}
                    self._conversion_parameters[
                        series_uid] = conversion_parameter
                except IOError:
                    print(
                        'Cannot read {}'.format(preprocessing_info_file))
            else:
                raise IOError('{} does not exist'.format(
                    preprocessing_info_file))

    def compute_cropped_voxel_coordinates(self):
        for series_uid, coords in self._coordinates.iteritems():
            for coord in coords:
                coord.update({world.format(dim): [
                    c * self._conversion_parameters[series_uid][
                        'cropped_grid_shape'][
                        '{}'.format(dim)] + \
                    self._conversion_parameters[series_uid]['extendbox_origin'][
                        '{}'.format(dim)] for c in
                    coord[voxel.format(dim)]] for dim in dimensions})

    def compute_world_coordinates(self):
        for series_uid, coords in self._coordinates.iteritems():
            for coord in coords:
                for dim in dimensions:
                    coord[world.format(dim)] = [
                        c * self._conversion_parameters[series_uid][
                            'resampled_spacing'][
                            '{}'.format(dim)] + \
                        self._conversion_parameters[series_uid][
                            'original_origin'][
                            '{}'.format(dim)] for c in
                        coord[world.format(dim)]]
                    start_and_end = coord[world.format(dim)]
                    coord[world.format(dim)] = [start_and_end[0],
                                                sum(start_and_end) * 0.5,
                                                start_and_end[1]]

    def output_json(self):
        with open(os.path.join(self._output_path,
                               'detected_nodules_in_world_and_voxel_coordinates.json'),
                  'w') as json_handle:
            json.dump(self._coordinates, json_handle, indent=4)
