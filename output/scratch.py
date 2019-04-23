import json
import os

input_dict = {}

if os.path.exists('../prep_result/preprocessing_info.txt'):
    with open('../prep_result/preprocessing_info.txt', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            key = line.split('=')[0]
            values = line.split('=')[1]
            coord = values.split(',')
            input_dict[key] = {'x': float(coord[0]), 'y': float(coord[1]),
                               'z': float(coord[2].strip('\n'))}
else:
    print("File does not exist")
print(input_dict)
input_dict['resampled_spacing'] = {'x': 1.0, 'y': 1.0,
                     'z': 1.0}  # harcoded to be this, no need to output

# cropped_grid_shape = {'x': 328.0, 'y': 240.0, 'z': 307.0}
# original_origin = {'x': -216.0, 'y': -148.0, 'z': 224.5}
# original_spacing = {'x': 0.78125, 'y': 0.78125, 'z': 1.0}
# resampled_spacing = {'x': 1.0, 'y': 1.0,
#                      'z': 1.0}  # harcoded to be this, no need to output
# extendbox_origin = {'x': 46.0, 'y': 112.0, 'z': 21.0}
#
boundingbox_min = {'x': [], 'y': [], 'z': []}
boundingbox_max = {'x': [], 'y': [], 'z': []}
with open("crop_rects.json") as f:
    cropped_rects = json.load(f)
for key, items in cropped_rects.iteritems():
    for boundingbox in items:
        for dim, min_and_max in boundingbox.iteritems():
            boundingbox_min[dim].append(float(min_and_max[0]))
            boundingbox_max[dim].append(float(min_and_max[1]))

print("Original bounding box")
print(boundingbox_min)
print(boundingbox_max)

for dim, value in input_dict['extendbox_origin'].iteritems():
    for index, min_coord in enumerate(
            boundingbox_min[dim]):
        boundingbox_min[dim][index] = min_coord * input_dict['cropped_grid_shape'][
            dim] + value
for dim, value in input_dict['extendbox_origin'].iteritems():
    for index, max_coord in enumerate(
            boundingbox_max[dim]):
        boundingbox_max[dim][index] = max_coord * input_dict['cropped_grid_shape'][
            dim] + value

print(boundingbox_min)
print(boundingbox_max)

for dim, value in input_dict['original_origin'].iteritems():
    for index, min_coord in enumerate(
            boundingbox_min[dim]):
        boundingbox_min[dim][index] = min_coord * input_dict['resampled_spacing'][dim] + value

for dim, value in input_dict['original_origin'].iteritems():
    for index, max_coord in enumerate(
            boundingbox_max[dim]):
        boundingbox_max[dim][index] = max_coord * input_dict['resampled_spacing'][dim] + value

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
with open("converted_voxel_to_world.txt", "w") as output:
    output.write(output_string)
# [(74.0 115.0 398.5)  # 5, (-65.0 45.0 494.5) #4, (-53.0 169.0 267.5) #3, (-136.0 17.0 279.5) #2, (-156.0 53.0 263.5) #1]
# [(74.0 115.0 398.5) #5, (-65.0 45.0 494.5) #4, (-53.0 169.0 267.5) #3, (-136.0 17.0 279.5) #2, (-156.0 53.0 263.5) #1]
#
# # TODO: Cleanup
# # TODO: Output JSON File format -
# # rel_x, rel_y, wc_x box start, wc_x box end, center
