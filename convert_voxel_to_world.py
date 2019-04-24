import json
import os
import copy

def convert_voxel_to_world(prep_folder, name, crop_rects_json_path,
                           output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    input_dict = {}
    preprocessing_info_file_name = os.path.join(prep_folder,
                                                '{}_preprocessing_info.txt'.format(
                                                    name))
    if os.path.exists(preprocessing_info_file_name):
        with open(preprocessing_info_file_name, 'r') as handle:
            lines = handle.readlines()
            for line in lines:
                key = line.split('=')[0]
                values = line.split('=')[1]
                coord = values.split(',')
                input_dict[key] = {'x': float(coord[0]), 'y': float(coord[1]),
                                   'z': float(coord[2].strip('\n'))}
    else:
        print("{} does not exist".format(preprocessing_info_file_name))
        return
    print(input_dict)
    input_dict['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                                       'z': 1.0}  # harcoded to be this, no need to output

    boundingbox_min = {'x': [], 'y': [], 'z': []}
    boundingbox_max = {'x': [], 'y': [], 'z': []}
    with open(crop_rects_json_path) as f:
        cropped_rects = json.load(f)
    for key, items in cropped_rects.iteritems():
        for boundingbox in items:
            for dim, min_and_max in boundingbox.iteritems():
                boundingbox_min[dim].append(float(min_and_max[0]))
                boundingbox_max[dim].append(float(min_and_max[1]))

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
