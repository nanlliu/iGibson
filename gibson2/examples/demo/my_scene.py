import cv2
import os

import numpy as np
import json
import pybullet as p

import random
import gibson2
from gibson2.robots.locobot_robot import Locobot
from gibson2.utils.utils import parse_config
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator

directions = {'right': np.array([0, 0, -0.7071, 0.7071]),
              'left': np.array([0, 0, 0.7071, 0.7071]),
              'behind': np.array([0, 0, 0, 1]),
              'front': np.array([0, 0, 1, 0])}

sofa_pos = {'right': np.array([3.9681901931762695, 0.3906211853027344, 0.2956485152244568]),
            'left': np.array([1.4681901931762695, 0.3906211853027344, 0.2956485152244568]),
            'behind': np.array([2.7181901931762695, 2.1406211853027344, 0.2956485152244568]),
            'front': np.array([2.4681901931762695, -0.8593788146972656, 0.2956485152244568])}

left_boundary = 1.5
right_boundary = 4
front_boundary = 0
back_boundary = 2

obj_keys = ['sofa_25', 'coffee_table_26', 'bottom_cabinet_28', 'floor_lamp_27']
obj_directions = ('right', 'left', 'front', 'behind')

save_path = '/home/ubuntu/igibson_dataset/'

all_possible_materials = {'fabric': 0, 'leather': 1, 'metal': 2, 'plastic': 3,
                          'paint': 4, 'wood': 5, 'paper': 6, 'chipboard': 7, 'none': 8}

all_possible_colors = {'brown': 0, 'gray': 1, 'iris': 2, 'purple': 3,
                       'white': 4, 'black': 5, 'baby blue': 6, 'drifted gray': 7, 'blue': 8, 'none': 9}

all_possible_shapes = {'sofa_25': 0, 'coffee_table_26': 1, 'bottom_cabinet_28': 2,
                       'floor_lamp_27': 3, 'none': 4}

all_possible_relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3, 'none': 4}


def print_scene(scene):
    # using scene to print out info
    for rel, indices in scene['relationships'].items():
        for i, other_indices in enumerate(indices):
            obj1 = scene['objects'][i]
            if not obj1:
                continue
            obj1_name = ' '.join([obj1['color'], obj1['material'], obj1['shape']])
            for j in other_indices:
                obj2 = scene['objects'][j]
                if not obj2:
                    continue
                obj2_name = ' '.join([obj2['color'], obj2['material'], obj2['shape']])
                print(obj2_name, rel, obj1_name)


def get_euclidean_distance(pos1, pos2):
    return np.linalg.norm(pos1[:2] - pos2[:2])


def compute_relationships(locations, epsilon=0.5):
    relationships = {}
    
    for direction in ['left', 'right', 'front', 'behind']:
        relationships[direction] = []
        for i, loc1 in enumerate(locations):
            # when it is sofa, we need to have bigger gap
            # to measure directions since sofa is large
            if loc1 is None:
                relationships[direction].append(list())
                continue
            related = set()
            x1, y1 = loc1[0], loc1[1]
            
            for j, loc2 in enumerate(locations):
                if loc2 is None:
                    continue
                x2, y2 = loc2[0], loc2[1]
                if i == j:
                    continue
                # sofa
                if i == 0 or j == 0:
                    new_epsilon = 2 * epsilon
                else:
                    new_epsilon = epsilon
                    
                if direction == 'front':
                    if x2 < x1 - new_epsilon:
                        related.add(j)
                elif direction == 'behind':
                    if x2 > x1 + new_epsilon:
                        related.add(j)
                elif direction == 'right':
                    if y2 < y1 - new_epsilon:
                        related.add(j)
                elif direction == 'left':
                    if y2 > y1 + new_epsilon:
                        related.add(j)
                        
            relationships[direction].append(sorted(related))
    
    return relationships


def print_relations(relationships):
    for rel, indices in relationships.items():
        for i, other_indices in enumerate(indices):
            for j in other_indices:
                print(obj_keys[j], rel, obj_keys[i])
        
        
def change_sofa_position(obj_ids, obj_pos):
    new_pos = []
    
    direction = np.random.choice(obj_directions)
    
    for k, obj_id in enumerate(obj_ids):
        pos1 = obj_pos[k]
        
        if direction == 'right':
            # if it is sofa object, otherwise pillow
            if k == 1:
                pos1 = np.array([pos1[0] + 1.25, pos1[1] - 1.75, pos1[2]])
            else:
                pos1 = np.array([pos1[0] + 100, pos1[1] - 100, pos1[2]])
        elif direction == 'left':
            # if it is sofa object, otherwise pillow
            if k == 1:
                pos1 = np.array([pos1[0] - 1.5, pos1[1] - 1.75, pos1[2]])
            else:
                pos1 = np.array([pos1[0] - 100, pos1[1] - 100, pos1[2]])
        elif direction == 'behind':
            if k == 1:
                pos1 = np.array([pos1[0] - 0.25, pos1[1], pos1[2]])
            else:
                pos1 = np.array([pos1[0] + 100, pos1[1] - 100, pos1[2]])
        elif direction == 'front':
            if k == 1:
                pos1 = np.array([pos1[0] - 0.25, pos1[1] - 3, pos1[2]])
            else:
                pos1 = np.array([pos1[0] + 100, pos1[1] - 100, pos1[2]])
        else:
            raise NotImplementedError
        
        new_pos.append(pos1)
        
        p.resetBasePositionAndOrientation(obj_id,
                                          pos1,
                                          directions[direction])
    # return new position of the sofa
    return new_pos[1], direction
    
    
def change_other_position(obj_ids, obj_pos, obj_locations, epsilon=1.):
    assert len(obj_ids) == 1
    
    pos = obj_pos[0]
    obj_id = obj_ids[0]
    
    direction = np.random.choice(obj_directions)
    
    while True:
        x, y = np.random.uniform(left_boundary, right_boundary), \
               np.random.uniform(front_boundary, back_boundary)
        
        new_pos = np.array([x, y, pos[2]])
        
        is_valid = True
        
        for i, other_obj in enumerate(obj_locations):
            if other_obj is not None:
                if i == 0:
                    new_epsilon = 1.5 * epsilon
                else:
                    new_epsilon = epsilon
                    
                if get_euclidean_distance(new_pos, other_obj) >= new_epsilon:
                    continue
                else:
                    is_valid = False
                    break
        
        if is_valid:
            break
            
    p.resetBasePositionAndOrientation(obj_id, new_pos, directions['right'])
    
    return new_pos, direction


def change_lamp_position(obj_ids, obj_pos, obj_locations, epsilon=1.):
    assert len(obj_ids) == 1
    
    pos = obj_pos[0]
    obj_id = obj_ids[0]
    
    direction = np.random.choice(obj_directions)
    
    while True:
        x, y = np.random.uniform(left_boundary + 0.25, right_boundary - 0.25), \
               np.random.uniform(front_boundary - 0.5, back_boundary - 0.25)
        
        new_pos = np.array([x, y, pos[2]])
        
        is_valid = True
        
        for i, other_obj in enumerate(obj_locations):
            
            if other_obj is not None:
                
                if i == 0:
                    new_epsilon = 2 * epsilon
                else:
                    new_epsilon = epsilon
                    
                if get_euclidean_distance(new_pos, other_obj) >= new_epsilon:
                    continue
                else:
                    is_valid = False
                    break
        
        if is_valid:
            break
    
    p.resetBasePositionAndOrientation(obj_id, new_pos, directions['right'])
    
    return new_pos, direction


def extract_relation(relationships, idx1, idx2):
    all_rel = []
    for rel, indices in relationships.items():
        for curr_index, index in enumerate(indices):
            if curr_index == idx2 and idx1 in index:
                all_rel.append(rel)
    if all_rel:
        return random.choice(all_rel)
    else:
        return None
    

def convert_label(json_scene, possible_object_indices):
    if len(possible_object_indices) == 1:
        idx1, idx2 = possible_object_indices[0], -1
        relation = 'none'
    else:
        idx1, idx2 = np.random.choice(possible_object_indices, size=2, replace=False)
        assert idx1 != idx2
        relation = extract_relation(json_scene['relationships'], idx1, idx2)
        assert relation is not None
        
    obj1 = json_scene['objects'][idx1]
    obj1_color = obj1['color']
    obj1_shape = obj1['shape']
    obj1_material = obj1['material']
    
    if idx2 == -1:
        obj2_color = 'none'
        obj2_shape = 'none'
        obj2_material = 'none'
    else:
        obj2 = json_scene['objects'][idx2]
        obj2_color = obj2['color']
        obj2_shape = obj2['shape']
        obj2_material = obj2['material']
    
    # print('converting...')
    # print(' '.join([obj1_color, obj1_material, obj1_shape]),
    #       relation,
    #       ' '.join([obj2_color, obj2_material, obj2_shape]))
    
    label = [all_possible_colors[obj1_color],
             all_possible_materials[obj1_material],
             all_possible_shapes[obj1_shape],
             all_possible_relations[relation],
             all_possible_colors[obj2_color],
             all_possible_materials[obj2_material],
             all_possible_shapes[obj2_shape]
             ]
    
    # print(label)
    
    return label
    

def main():
    ##############################
    # Set up simulator and scene #
    ##############################
    s = Simulator(mode='headless',
                  image_width=128,
                  image_height=128,
                  device_idx=0,
                  render_to_tensor=True)
    
    load_object_categories = ['sofa', 'coffee_table', 'bottom_cabinet', 'floor_lamp']

    scene = InteractiveIndoorScene('Ihlen_1_int',
                                   texture_randomization=True,
                                   object_randomization=False,
                                   load_room_types=['living_room'],
                                   load_object_categories=load_object_categories)
    
    s.import_ig_scene(scene)
    s.renderer.set_fov(100)
    
    ##############################
    # Extract object information #
    ##############################
    
    obj_pos, obj_orn, obj_ids = {}, {}, {}

    keys = ['sofa_25', 'coffee_table_26', 'bottom_cabinet_28', 'floor_lamp_27']
    
    for obj_key in keys:
        curr_obj_pos, curr_obj_orn, curr_obj_ids = [], [], []
        
        for i, body_id in enumerate(scene.objects_by_name[obj_key].body_ids):
            pos, orn = p.getBasePositionAndOrientation(body_id)
            
            curr_obj_pos.append(np.array(pos))
            curr_obj_orn.append(np.array(orn))
            curr_obj_ids.append(body_id)
        
        obj_pos[obj_key] = curr_obj_pos
        obj_orn[obj_key] = curr_obj_orn
        obj_ids[obj_key] = curr_obj_ids
    
    # ##############################
    # #  Render randomized images  #
    # ##############################
    
    for i in range(0, 100000):

        info = scene.randomize_texture()

        obj_locations = []
        
        key_indices = np.random.choice(4, size=np.random.randint(low=1, high=4, size=1), replace=False)
        
        curr_keys = [None] * 4
        for key_index in key_indices:
            curr_keys[key_index] = obj_keys[key_index]
        
        for j, key in enumerate(obj_keys):
            curr_obj_pos = obj_pos[key]
            curr_obj_ids = obj_ids[key]
            
            if curr_keys[j] is not None:
                if key == 'sofa_25':
                    new_pos, new_orn = change_sofa_position(curr_obj_ids, curr_obj_pos)
                elif key == 'floor_lamp_27':
                    new_pos, new_orn = change_lamp_position(curr_obj_ids, curr_obj_pos, obj_locations, epsilon=1)
                else:
                    new_pos, new_orn = change_other_position(curr_obj_ids, curr_obj_pos, obj_locations, epsilon=1)
            else:
                # remove object from the scene
                for obj_idx, obj_id in enumerate(curr_obj_ids):
                    x, y, z = curr_obj_pos[obj_idx]
                    p.resetBasePositionAndOrientation(obj_id, (-4, 0, z), directions['left'])
                
                # make sure any distance from other objects is valid
                new_pos = None

            # add object's new position for relation calculation
            obj_locations.append(new_pos)

        # after move all objects and render
        for _ in range(2):
            s.step()

        # render image
        # left to right shifting
        y = np.random.uniform(-1, 1)
        camera_pose = np.array([0.4, 0.5 + y, 1.8])
        
        view_direction = np.array([1.2 - np.abs(y / 6.), 0 - y / 2., -1.1])
        
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        
        im = s.renderer.render(modes=('rgb'))[0] * 255.
        # calculate relationships
        relationships = compute_relationships(obj_locations, epsilon=0.5)

        # save scene and image
        json_scene = {'relationships': relationships, 'objects': []}
        for index, key in enumerate(curr_keys):
            if key is not None:
                object_dict = {'shape': key, 'material': info[key]['material'], 'color': info[key]['color']}
                json_scene['objects'].append(object_dict)
            else:
                json_scene['objects'].append(None)

        label = convert_label(json_scene, key_indices)
        json_scene['label'] = label
        
        # print_relations(relationships)
        # print_scene(json_scene)

        # cv2.imwrite(f'test-{i}.png', im)
        
        json_path = os.path.join(save_path, 'scenes_1/igibson_scene_{:06}.json'.format(i))
        with open(json_path, 'w') as f:
            json.dump(json_scene, f)

        image_path = os.path.join(save_path, 'images_1/igibson_image_{:06}.png'.format(i))
        cv2.imwrite(image_path, im)

        print(json_path)
        print(image_path)

    s.disconnect()


if __name__ == '__main__':
    main()
