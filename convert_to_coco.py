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

def reproject_to_3d(im_coords, K, z):
    im_coords = np.stack([im_coords[:,0], im_coords[:,1]],axis=1)
    im_coords = np.hstack((im_coords, np.ones((im_coords.shape[0],1))))
    projected = np.dot(np.linalg.inv(K), im_coords.T).T
    projected[:, 0] *= z
    projected[:, 1] *= z
    projected[:, 2] *= z
    return projected

def convert_samples(base_path, set_type="training"):
    from utils.model import recover_root, get_focal_pp, split_theta

    # load annotations
    db_data_anno = load_db_annotation(base_path, set_type)
    output = {
        "images": [],
        "annotations": [],
        "categories": [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person'
        }]
    }
    
    # Sticking with gs for now
    version = "gs"
    # iterate over all samples
    for idx in range(db_size(set_type)):
        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        poses, shapes, uv_root, scale = split_theta(mano)
        focal, pp = get_focal_pp(K)
        xyz_root = recover_root(uv_root, scale, focal, pp)
        os.path.join(
            base_path, set_type, 'rgb',
            '%08d.jpg' % sample_version.map_id(idx, version)
        )

        uv *= (float(256)/224)
        xyz = reproject_to_3d(uv, K, xyz[:, 2])
                

        output["images"].append({
            "id": idx,
            "width": 256,
            "height": 256,
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
            "joint_img": uv.tolist(),
            "joint_valid": np.ones(21).tolist(),
            "hand_type": "right",
            "joint_cam": (xyz * 1000).tolist(),
            "bbox": get_bbox(uv)
        })

    with open('freihand_' + set_type + '.json', 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')

    args = parser.parse_args()
    #convert_eval_samples(args.base_path, num2show=args.num2show)

    convert_samples(
        args.base_path,
        "training"
    )

    convert_samples(
        args.base_path,
        "evaluation"
    )
