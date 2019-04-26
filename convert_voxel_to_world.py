import json
import os

world = 'world_{}'
voxel = 'rel_voxel_{}'
dimensions = ['x', 'y', 'z']


class ConvertVoxelToWorld(object):
    def __init__(self, prep_folder, series_uid, cropped_rects,
                 output_path):
        self._conversion_parameters = {}
        self._coordinates = []
        self._preprocessing_info_file = os.path.join(prep_folder,
                                                     '{}_preprocessing_info.txt'.format(
                                                         series_uid))
        self._list_of_cropped_rects = cropped_rects
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
                'original_origin, extendbox_origin, cropped_grid_shape, resampled_spacing must be present in {}'.format(
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
                print('Cannot read {}'.format(self._preprocessing_info_file))
        else:
            raise IOError('{} does not exist'.format(
                self._preprocessing_info_file))
        self._conversion_parameters['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                                                            'z': 1.0}

    def get_relative_voxel_coordinates(self):
        for items in self._list_of_cropped_rects:
            for boundingbox in items:
                coordinate = {world.format(dim): [] for dim in dimensions}
                coordinate.update(
                    {voxel.format(dim): min_and_max for dim, min_and_max
                     in boundingbox.iteritems()})
                self._coordinates.append(coordinate)

    def compute_cropped_voxel_coordinates(self):
        for coord in self._coordinates:
            for dim in dimensions:
                coord[world.format(dim)] = [
                    c * self._conversion_parameters['cropped_grid_shape'][
                        '{}'.format(dim)] + \
                    self._conversion_parameters['extendbox_origin'][
                        '{}'.format(dim)] for c in
                    coord[voxel.format(dim)]]

    def compute_world_coordinates(self):
        for coord in self._coordinates:
            for dim in dimensions:
                coord[world.format(dim)] = [
                    c * self._conversion_parameters['resampled_spacing'][
                        '{}'.format(dim)] + \
                    self._conversion_parameters['original_origin'][
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
            json.dump(self._coordinates, json_handle)
