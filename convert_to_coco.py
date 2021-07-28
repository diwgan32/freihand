# Diwakar Ganesan 07/27/2021
# Convert Freihand JSON files to
# COCO format for use in Interhand2.6M model

from __future__ import print_function, unicode_literals
import matplotlib  
matplotlib.use('TkAgg')
import json
import matplotlib.pyplot as plt
import argparse

from utils.fh_utils import *


def convert_training_samples(base_path):
    from utils.model import recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all

    # load annotations
    db_data_anno = load_db_annotation(base_path, 'training')
    output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    set_name = "training"

    # Sticking with gs for now
    version = "gs"
    # iterate over all samples
    for idx in range(db_size('training')):
        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        poses, shapes, uv_root, scale = split_theta(mano)
        focal, pp = get_focal_pp(K)
        xyz_root = recover_root(uv_root, scale, focal, pp)
        os.path.join(
            base_path, set_name, 'rgb',
            '%08d.jpg' % sample_version.map_id(idx, version)
        )
        output["images"].append({
            "id": idx,
            "width": 224,
            "height": 224,
            "file_name": '%08d.jpg' % sample_version.map_id(idx, version),
            "camera_param": {
                "focal": [K[0][0], K[1][1]],
                "princpt": [K[0][2], K[1][2]]
            }
        })

        output["annotations"].append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "is_crowd": 0,
            "joint_img": uv,
            "joint_valid": np.ones(21).tolist(),
            "hand_type": "right",
            "joint_cam": xyz,
            "bbox": get_bbox(uv)
        })
    with open('freihand_training.json', 'w') as f:
        json.dump(output, f)

def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in  range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')

    args = parser.parse_args()
    #convert_eval_samples(args.base_path, num2show=args.num2show)

    convert_training_samples(
        args.base_path
    )

