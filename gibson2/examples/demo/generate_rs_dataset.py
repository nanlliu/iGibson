import os
import json
import numpy as np
import pybullet as p

from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator
from gibson2.utils.mesh_util import normalize, quat2rotmat, mat2xyz, xyzw2wxyz
from gibson2.utils.utils import rotate_vector_3d, get_rpy_from_transform

from skimage.io import imsave
from skimage import img_as_ubyte

import torch
from torchvision.utils import make_grid, save_image


loaded_objects = ['walls', 'floors', 'ceilings', 'sofa', 'floor_lamp',
                  'coffee_table', 'chair', 'counter', 'window',
                  'bottom_cabinet', 'table', 'stool']

manipulated_objects = ['coffee_table', 'bottom_cabinet', 'stool', 'sofa']

background_objects = list(set(loaded_objects) - set(manipulated_objects))

facing_directions = {'right': np.array([0, 0, 0.70786583, 0.70634692]),
                     'left': np.array([0, 0, -0.70786583, 0.70634692]),
                     'behind': np.array([0, 0, 0, 1]),
                     'front': np.array([0, 0, 1, 0])}

sofa_pos = {'right': np.array([-1.51745653, -1.72151279, 0.39529318]),
            'left': np.array([1.51745653, -1.72151279, 0.39529318]),
            'behind': np.array([0, -0.5, 0.39529318]),
            'front': np.array([0, -2.9, 0.39529318])}

all_possible_colors = {'blue': 0, 'gray': 1, 'red': 2, 'maple': 3, 'garden walnut': 4, 'none': 5}
all_possible_shapes = {'sofa_2': 0, 'coffee_table_5': 1, 'bottom_cabinet_0': 2, 'stool_4': 3, 'none': 4}
all_possible_materials = {'fabric': 0, 'leather': 1, 'wood': 2, 'none': 3}
all_possible_relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3, 'none': 4}


def project_u_on_v(u, v):
    return v * np.dot(u, v) / np.dot(v, v)


def cam_vectors(simulator):
    transform_matrix = simulator.renderer.V.copy()
    r, p, y = get_rpy_from_transform(transform_matrix)

    cam_left = np.array([-1, 0, 0])
    cam_behind = np.array([0, -1, 0])

    cam_left = rotate_vector_3d(cam_left, r, p, y)
    cam_behind = rotate_vector_3d(cam_behind, r, p, y)

    plane_normal = np.array([0, 0, 1])

    plane_left = normalize(cam_left - project_u_on_v(u=cam_left, v=plane_normal))
    plane_behind = normalize(cam_behind - project_u_on_v(u=cam_behind, v=plane_normal))
    plane_right = -plane_left
    plane_front = -plane_behind

    return plane_left, plane_right, plane_front, plane_behind


def compute_all_relationships(scene_struct, eps=0.5):
    """
    Copy and modify from https://github.com/facebookresearch/clevr-dataset-gen/
    blob/f0ce2c81750bfae09b5bf94d009f42e055f2cb3a/image_generation/render_images.py#L448

    Computes relationships between all pairs of objects in the scene.
    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, (key1, obj1) in enumerate(scene_struct['objects'].items()):
            # obj1 could be empty dict
            if not obj1:
                all_relationships[name].append([])
                continue
            coords1 = obj1['pos']
            related = set()
            for j, (key2, obj2) in enumerate(scene_struct['objects'].items()):
                # obj2 could be empty dict
                if i == j or not obj2:
                    continue
                coords2 = obj2['pos']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if 'sofa' in key1 or 'sofa' in key2:
                    if dot > 2.5 * eps:
                        related.add(j)
                elif dot > 1.5 * eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def extract_pos_ori_and_body_ids(objects_by_name):
    pos_ori = {}
    body_ids = {}
    obj_sizes = {}

    for key, obj in objects_by_name.items():
        assert len(obj.body_ids) == 1

        # skip background objects and the tv stand
        if 'coffee_table' not in key and any(s in key for s in background_objects) \
                or key == 'bottom_cabinet_13':
            continue

        body_id = obj.body_ids[0]
        object_pos_ori = {}

        pos, orn = p.getBasePositionAndOrientation(body_id)
        object_pos_ori['pos'] = np.array(pos)
        object_pos_ori['orn'] = np.array(orn)

        pos_ori[key] = object_pos_ori
        body_ids[key] = body_id
        obj_sizes[key] = obj.bounding_box[0]

    return pos_ori, body_ids, obj_sizes


def print_relationships(scene_struct):
    relationships = scene_struct['relationships']
    directions = list(relationships.keys())
    object_names = list(scene_struct['objects'].keys())

    for directions_type in directions:
        relations = relationships[directions_type]
        for curr_obj_idx in range(len(relations)):
            for other_obj_idx in relations[curr_obj_idx]:
                print(object_names[other_obj_idx], directions_type, object_names[curr_obj_idx])


def randomize_objects(pos_ori, curr_keys, body_ids, coordinates_range, dist=1):

    new_pos_ori = {}
    x_min, x_max, y_min, y_max = coordinates_range

    for key in ['sofa_2', "stool_4", "coffee_table_5", "bottom_cabinet_0"]:
        body_id = body_ids[key]

        # remove objects from the scene if they are not selected
        if key not in curr_keys:
            new_pos_ori[key] = {}
            x, y, z = pos_ori[key]['pos']
            if key == 'bottom_cabinet_0':
                # leave in a place where it is stable
                p.resetBasePositionAndOrientation(body_id, (-3.2, -0.4, z), facing_directions['left'])
            else:
                p.resetBasePositionAndOrientation(body_id, (-5, -0.4, z), facing_directions['left'])
            continue

        while True:
            no_collision = True

            if key == 'sofa_2':
                new_facing_direction = np.random.choice(['left', 'right', 'front', 'behind'])
                new_orn = facing_directions[new_facing_direction]
                new_pos = sofa_pos[new_facing_direction]
            elif key == 'coffee_table_5':
                new_facing_direction = np.random.choice(['left', 'right'])
                new_orn = facing_directions[new_facing_direction]
                new_pos = np.array(
                    [np.random.uniform(x_min, x_max),
                     np.random.uniform(y_min, y_max),
                     pos_ori[key]['pos'][2]]
                )
            else:
                new_pos = np.array(
                    [np.random.uniform(x_min, x_max),
                     np.random.uniform(y_min, y_max),
                     pos_ori[key]['pos'][2]]
                )
                new_orn = facing_directions['front']

            p.resetBasePositionAndOrientation(body_id, new_pos, new_orn)

            # check distance
            for curr_key, curr_pos_ori in new_pos_ori.items():
                if not curr_pos_ori:
                    continue
                dx, dy = new_pos[0] - curr_pos_ori['pos'][0], new_pos[1] - curr_pos_ori['pos'][1]
                center_dist = np.sqrt(dx ** 2 + dy ** 2)

                if curr_key == 'sofa_2':
                    new_dist = 1.5 * dist
                else:
                    new_dist = 1.2 * dist

                if center_dist < new_dist:
                    no_collision = False
                    break

            if no_collision:
                new_pos_ori[key] = {}
                new_pos_ori[key]['pos'] = new_pos
                new_pos_ori[key]['orn'] = new_orn
                break

    return new_pos_ori


def extract_relation(relationships, idx1, idx2):
    all_rel = []
    for rel, indices in relationships.items():
        for curr_index, index in enumerate(indices):
            if curr_index == idx2 and idx1 in index:
                all_rel.append(rel)
    if all_rel:
        return np.random.choice(all_rel)
    else:
        return None


def main():
    render_to_tensor = False

    s = Simulator(mode='headless',
                  image_width=256,
                  image_height=256,
                  device_idx=0,
                  render_to_tensor=render_to_tensor)

    scene = InteractiveIndoorScene(
        'Rs_int',
        texture_randomization=True,
        object_randomization=False,
        load_room_types=['living_room'],
        load_object_categories=loaded_objects)

    s.import_ig_scene(scene)

    camera = np.array([-0.1, -3.5, 2.3])
    view_direction = np.array([0, 0.7, -0.7])
    s.renderer.set_camera(camera, camera + view_direction, [0, 0, 1])

    scene_struct = {
        'directions': {},
        'objects': {}
    }

    left, right, front, behind = cam_vectors(s)

    scene_struct['directions']['left'] = tuple(left)
    scene_struct['directions']['right'] = tuple(right)
    scene_struct['directions']['front'] = tuple(behind)
    scene_struct['directions']['behind'] = tuple(front)

    pos_ori, body_ids, object_sizes = extract_pos_ori_and_body_ids(scene.objects_by_name)
    object_names = ['sofa_2', "stool_4", "coffee_table_5", "bottom_cabinet_0"]

    scene_struct['objects'] = pos_ori
    relationships = compute_all_relationships(scene_struct)
    scene_struct['relationships'] = relationships

    x_min = -1.25
    x_max = 1.25
    y_min = -2.9
    y_max = -0.5
    
    images = []

    for i in range(50000):
        jitter = np.random.uniform(-0.8, 0.8, size=1)

        camera = np.array([-0.1, -3.5, 2.3])
        camera[0] += jitter
        view_direction = np.array([0, 0.7, -0.7])
        view_direction[0] -= jitter / 3
        # view_direction[1] -= jitter / 2
        s.renderer.set_camera(camera, camera + view_direction, [0, 0, 1])

        info = scene.randomize_texture()

        object_keys_indices = sorted(np.random.choice(
            4, size=np.random.randint(low=1, high=4, size=1), replace=False))

        curr_keys = [None] * 4
        for key_index in object_keys_indices:
            curr_keys[key_index] = object_names[key_index]

        object_keys = set([object_names[i] for i in object_keys_indices])

        curr_pos_ori = randomize_objects(pos_ori, object_keys, body_ids, (x_min, x_max, y_min, y_max))

        scene_struct['objects'] = curr_pos_ori
        relationships = compute_all_relationships(scene_struct)
        scene_struct['relationships'] = relationships

        # print_relationships(scene_struct)

        for j in range(3):
            s.step()

        # save scene and image
        json_scene = {'relationships': relationships, 'objects': []}
        for key in curr_keys:
            if key is not None:
                object_dict = {'shape': key, 'material': info[key]['material'], 'color': info[key]['color']}
                json_scene['objects'].append(object_dict)
            else:
                json_scene['objects'].append(None)

        im = s.renderer.render(modes=('rgb'))[0]

        save_path = '/home/nan/data/igibson_50000_dataset'

        # imsave(f'test-{i}.png', im.cpu().numpy() if render_to_tensor else img_as_ubyte(im))

        json_path = os.path.join(save_path, 'scenes/igibson_scene_{:06}.json'.format(i))
        with open(json_path, 'w') as f:
            json.dump(json_scene, f)

        image_path = os.path.join(save_path, 'images/igibson_image_{:06}.png'.format(i))
        imsave(image_path, im.cpu().numpy() if render_to_tensor else img_as_ubyte(im))

        # print(image_path)
        # images.append(im)

    # grid = make_grid(torch.from_numpy(np.array(images).transpose((0, 3, 1, 2))), nrow=3)
    # save_image(grid, 'grid.png')

    s.disconnect()


if __name__ == '__main__':
    main()
