import json
import os
import numpy as np
import argparse
from multiprocessing import Pool
from PIL import Image
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

data_dir = '/home/nan/data/igibson_dataset'


def convert_save_images(index):
    image_path = os.path.join(data_dir, 'images/igibson_image_{:06}.png'.format(index))
    im = Image.open(image_path).resize((128, 128), Image.ANTIALIAS)
    im.save(os.path.join(data_dir, 'converted_images/igibson_image_{:06}.png'.format(index)))


def extract_label(index):
    image_path = os.path.join(data_dir, 'images/igibson_image_{:06}.png'.format(index))
    scene_path = os.path.join(data_dir, 'scenes/igibson_scene_{:06}.json'.format(index))
    im = np.asarray(Image.open(image_path).resize((128, 128), Image.ANTIALIAS).convert('RGB'))
    with open(scene_path, 'r') as f:
        json_scene = json.load(f)
    label = np.array(json_scene['label'])
    return im, label


def extract_all_labels(num_images):
    indices = list(range(num_images))
    pool = Pool()
    results = pool.map(extract_label, indices)
    ims, labels = zip(*results)
    final_ims = np.stack(ims, axis=0)
    final_labels = np.stack(labels, axis=0)
    return final_ims, final_labels


def convert_all_images(num_images):
    indices = list(range(num_images))
    pool = Pool()
    pool.map(convert_save_images, indices)


def merge_image_sets():
    final_ims, final_labels = extract_all_labels(num_images=30000)
    save_path = os.path.join(data_dir, 'igibson_train_30000_128.npz')
    print('image size', final_ims.shape)
    print('label size', final_labels.shape)
    np.savez(save_path, ims=final_ims, labels=final_labels)
    print('save in', save_path)


def print_stats():
    all_possible_colors = {'blue': 0, 'gray': 1, 'red': 2, 'maple': 3, 'garden walnut': 4, 'none': 5}
    all_possible_shapes = {'sofa_2': 0, 'coffee_table_5': 1, 'bottom_cabinet_0': 2, 'stool_4': 3, 'none': 4}
    all_possible_materials = {'fabric': 0, 'leather': 1, 'wood': 2, 'none': 3}
    all_possible_relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3, 'none': 4}

    materials = list(all_possible_materials.keys())
    colors = list(all_possible_colors.keys())
    shapes = list(all_possible_shapes.keys())
    relations = list(all_possible_relations.keys())
    save_path = os.path.join(data_dir, 'igibson_train_30000_256.npz')
    data = np.load(save_path)['labels']
    for i, attribute_list in enumerate([colors, materials, shapes, relations]):
        for j, attribute in enumerate(attribute_list):
            print(attribute, np.sum(data[:, i] == j))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--img_index", type=int)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--relation", action="store_true")
    parser.add_argument("--print_relation_stats", action="store_true")
    parser.add_argument("--convert", action="store_true")

    args = parser.parse_args()
    if args.merge:
        merge_image_sets()
    if args.stats:
        print_stats()

    if args.convert:
        convert_all_images(30000)
