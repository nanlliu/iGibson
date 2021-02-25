import cv2
import os

import numpy as np
import json
import pybullet as p

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

obj_directions = ['right', 'left', 'front', 'behind']

# sofa
# fabric {'diffuse': 76, 'metallic': None, 'roughness': 77, 'normal': 78}
# fabric {'diffuse': 70, 'metallic': None, 'roughness': 71, 'normal': 72}
# iris + fabric

# fabric {'diffuse': 58, 'metallic': None, 'roughness': 59, 'normal': 60}
# fabric {'diffuse': 58, 'metallic': None, 'roughness': 59, 'normal': 60}
# grey + fabric

# fabric {'diffuse': 70, 'metallic': None, 'roughness': 71, 'normal': 72}
# fabric {'diffuse': 64, 'metallic': None, 'roughness': 65, 'normal': 66}
# red + fabric

# leather {'diffuse': 88, 'metallic': None, 'roughness': 89, 'normal': 90}
# fabric {'diffuse': 82, 'metallic': None, 'roughness': 83, 'normal': 84}
# purple + leather

# leather {'diffuse': 106, 'metallic': None, 'roughness': 107, 'normal': 108}
# fabric {'diffuse': 76, 'metallic': None, 'roughness': 77, 'normal': 78}
# black + leather

# leather {'diffuse': 97, 'metallic': None, 'roughness': 98, 'normal': 99}
# fabric {'diffuse': 61, 'metallic': None, 'roughness': 62, 'normal': 63}
# white + leather

# coffee table
# chipboard {'diffuse': 124, 'metallic': None, 'roughness': 125, 'normal': 126}
# paper {'diffuse': 173, 'metallic': None, 'roughness': 174, 'normal': 175}
# paint {'diffuse': 170, 'metallic': None, 'roughness': 171, 'normal': 172}
# metal {'diffuse': 147, 'metallic': 148, 'roughness': 149, 'normal': 150}


# lamp
# metal {'diffuse': 55, 'metallic': 56, 'roughness': 57, 'normal': 58}
# plastic {'diffuse': 95, 'metallic': None, 'roughness': 96, 'normal': 97}
# paint {'diffuse': 107, 'metallic': None, 'roughness': 108, 'normal': 109}
# wood {'diffuse': 113, 'metallic': None, 'roughness': 114, 'normal': 115}

# cabinet
# wood {'diffuse': 55, 'metallic': None, 'roughness': 56, 'normal': 57}
# metal {'diffuse': 124, 'metallic': 125, 'roughness': 126, 'normal': 127}
# paint {'diffuse': 164, 'metallic': None, 'roughness': 165, 'normal': 166}


def get_euclidean_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def compute_relationships(locations, epsilon=0.5):
    relationships = {}
    
    for direction in ['left', 'right', 'front', 'behind']:
        relationships[direction] = []
        for i, loc1 in enumerate(locations):
            # when it is sofa, we need to have bigger gap
            # to measure directions since sofa is large
            related = set()
            x1, y1 = loc1[0], loc1[1]
                
            for j, loc2 in enumerate(locations):
                x2, y2 = loc2[0], loc2[1]
                if i == j:
                    continue
                # sofa
                if i == 0 or j == 0:
                    new_epsilon = 3 * epsilon
                else:
                    new_epsilon = epsilon
                    
                if direction == 'left':
                    if x2 < x1 - new_epsilon:
                        related.add(j)
                elif direction == 'right':
                    if x2 > x1 + new_epsilon:
                        related.add(j)
                elif direction == 'front':
                    if y2 < y1 - new_epsilon:
                        related.add(j)
                elif direction == 'behind':
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
            
    p.resetBasePositionAndOrientation(obj_id, new_pos, directions[direction])
    
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
    
    p.resetBasePositionAndOrientation(obj_id, new_pos, directions[direction])
    
    return new_pos, direction


def main():
    ##############################
    # Set up simulator and scene #
    ##############################
    s = Simulator(mode='headless',
                  image_width=256,
                  image_height=256,
                  device_idx=0,
                  render_to_tensor=True)
    
    load_object_categories = ['sofa', 'coffee_table', 'bottom_cabinet', 'floor_lamp']

    scene = InteractiveIndoorScene('Ihlen_1_int',
                                   texture_randomization=True,
                                   object_randomization=False,
                                   load_room_types=['living_room'],
                                   load_object_categories=load_object_categories)
    
    s.import_ig_scene(scene)
    
    camera_pose = np.array([2.5, -1.1, 2.2])
    view_direction = np.array([0, 0.6, -0.7])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(100)
    
    ##############################
    # Extract object information #
    ##############################
    
    obj_pos, obj_orn, obj_ids = {}, {}, {}

    obj_keys = ['sofa_25', 'coffee_table_26', 'bottom_cabinet_28', 'floor_lamp_27']
    
    for obj_key in obj_keys:
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
    
    for i in range(0, 1):

        new_materials, new_colors, names = scene.randomize_texture()

        obj_locations = []
        
        for key in obj_keys:
            curr_obj_pos = obj_pos[key]
            curr_obj_ids = obj_ids[key]

            if key == 'sofa_25':
                new_pos, new_orn = change_sofa_position(curr_obj_ids, curr_obj_pos)
            elif key == 'floor_lamp_27':
                new_pos, new_orn = change_lamp_position(curr_obj_ids, curr_obj_pos, obj_locations, epsilon=0.8)
            else:
                new_pos, new_orn = change_other_position(curr_obj_ids, curr_obj_pos, obj_locations, epsilon=1)
            
            # add object's new position for relation calculation
            obj_locations.append(new_pos)
        
        # debug
        relations = compute_relationships(obj_locations)
        print_relations(relations)
        for index in range(len(new_materials)):
            print(names[index], new_materials[index], new_colors[index])
        
        # after move all objects and render
        for _ in range(10):
            s.step()

        # render image
        im = s.renderer.render(modes=('rgb'))[0] * 255.
        cv2.imwrite(f'test-{i}.png', im)
                
    s.disconnect()


if __name__ == '__main__':
    main()
