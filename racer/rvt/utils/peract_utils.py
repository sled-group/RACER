# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].


import os
import pickle
import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

from peract_colab.arm.utils import stack_on_channel


# Contants
# TODO: Unclear about the best way to handle them
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DATA_FOLDER = "data"
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
# settings
NUM_LATENTS = 512  # PerceiverIO latents

# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT  = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def _preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        # front rgb torch.Size([3, 1, 3, 128, 128])
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])
        # front point cloud torch.Size([3, 1, 3, 128, 128])
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


def downsample_array(array, factor):
    if factor == 1:
        return array
    return array[1::factor, 1::factor]

def scale_intrinsics(intrinsics, factor):
    intrinsics[:2, :] /= factor
    return intrinsics

def get_stored_demo_aug(data_path, index, reference_demo):
    np.set_printoptions(precision=4, suppress=True)

    episode_path = os.path.join(data_path, str(index))
  
    # low dim pickle file
    with open(os.path.join(episode_path, "obs.pkl"), 'rb') as f:
        _obs = pickle.load(f)

    # with open(os.path.join(reference_data_path, EPISODE_FOLDER % index, LOW_DIM_PICKLE), 'rb') as f:
    #     reference_obs = pickle.load(f)
    reference_obs = reference_demo

    # sorted_id = sorted([int(k.split('_')[0]) for k in obs.keys()])
    # delete non-exist keys
    obs = {}
    for k in _obs.keys():
        if os.path.exists(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), "%s.png" % k)):
            obs[k] = _obs[k]
    
    for key in obs.keys():
        if "expert" in key or "dense" in key: # TODO image should also use original
            # replace certain attributes with reference data to avoid observed action failures
            # gripper open, gripper_joint_positions, ignore_collisions, gripper_pose
            key_id = int(key.split('_')[0])
            # if (obs[key].gripper_open != reference_obs[key_id].gripper_open or obs[key].ignore_collisions != reference_obs[key_id].ignore_collisions) and \
            #     key_id != sorted_id[0] and key_id != sorted_id[-1]:
            #     print("!!! Warning: gripper_open or ignore_collisions is different from reference data!!!")
            #     print(key)
            #     print(obs[key].gripper_open == reference_obs[key_id].gripper_open, obs[key].gripper_open, reference_obs[key_id].gripper_open, "gripper_open should be the same")
            #     print(obs[key].ignore_collisions == reference_obs[key_id].ignore_collisions, obs[key].ignore_collisions, reference_obs[key_id].ignore_collisions, "ignore_collisions should be the same")
            #     print("gripper_pose should be the same")
            #     print(obs[key].gripper_pose)
            #     print(reference_obs[key_id].gripper_pose)
            
            if "put_groceries_in_cupboard" in data_path:
                if obs[key].ignore_collisions != reference_obs[key_id].ignore_collisions \
                        and obs[key].ignore_collisions == 0:
                    pass
                else:
                    obs[key].ignore_collisions = reference_obs[key_id].ignore_collisions
            else:
                obs[key].ignore_collisions = reference_obs[key_id].ignore_collisions
            
            obs[key].gripper_open = reference_obs[key_id].gripper_open
            # obs[key].gripper_joint_positions = reference_obs[key_id].gripper_joint_positions
            obs[key].gripper_pose = reference_obs[key_id].gripper_pose

        obs[key].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), "%s.png" % key)))
        image_size = obs[key].front_rgb.shape[0]
        factor = image_size // IMAGE_SIZE
        obs[key].front_rgb = downsample_array(obs[key].front_rgb, factor)

        # # check whether there is huge difference between the two images
        # diff = np.mean(np.abs(obs[key].front_rgb - reference_obs[key_id].front_rgb), axis=-1)
        # print("front_rgb difference: ", np.max(diff), np.mean(diff), np.sum(diff >10))
        # # save two images together to compare
        # Image.fromarray(np.concatenate([obs[key].front_rgb, reference_obs[key_id].front_rgb], axis=1)).save("front_rgb.png")

        obs[key].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), "%s.png" % key)))
        obs[key].left_shoulder_rgb = downsample_array(obs[key].left_shoulder_rgb, factor)

        obs[key].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), "%s.png" % key)))
        obs[key].right_shoulder_rgb = downsample_array(obs[key].right_shoulder_rgb, factor)
        
        obs[key].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), "%s.png" % key)))
        obs[key].wrist_rgb = downsample_array(obs[key].wrist_rgb, factor)

        obs[key].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        obs[key].front_depth = downsample_array(obs[key].front_depth, factor)
        near = obs[key].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[key].front_depth = near + obs[key].front_depth * (far - near)

        obs[key].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        obs[key].left_shoulder_depth = downsample_array(obs[key].left_shoulder_depth, factor)
        near = obs[key].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_LS)]
        obs[key].left_shoulder_depth = near + obs[key].left_shoulder_depth * (far - near)

        obs[key].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        obs[key].right_shoulder_depth = downsample_array(obs[key].right_shoulder_depth, factor)
        near = obs[key].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_RS)]
        obs[key].right_shoulder_depth = near + obs[key].right_shoulder_depth * (far - near)

        obs[key].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        obs[key].wrist_depth = downsample_array(obs[key].wrist_depth, factor)
        near = obs[key].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[key].wrist_depth = near + obs[key].wrist_depth * (far - near)
        
        # scale down intrinsic matrix with factor
        scale_intrinsics(obs[key].misc['front_camera_intrinsics'], factor)
        scale_intrinsics(obs[key].misc['left_shoulder_camera_intrinsics'], factor)
        scale_intrinsics(obs[key].misc['right_shoulder_camera_intrinsics'], factor)
        scale_intrinsics(obs[key].misc['wrist_camera_intrinsics'], factor)
        

        obs[key].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[key].front_depth, 
                                                                                        obs[key].misc['front_camera_extrinsics'],
                                                                                        obs[key].misc['front_camera_intrinsics'])
        obs[key].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[key].left_shoulder_depth, 
                                                                                                obs[key].misc['left_shoulder_camera_extrinsics'],
                                                                                                obs[key].misc['left_shoulder_camera_intrinsics'])
        obs[key].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[key].right_shoulder_depth, 
                                                                                                obs[key].misc['right_shoulder_camera_extrinsics'],
                                                                                                obs[key].misc['right_shoulder_camera_intrinsics'])
        obs[key].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[key].wrist_depth, 
                                                                                            obs[key].misc['wrist_camera_extrinsics'],
                                                                                            obs[key].misc['wrist_camera_intrinsics'])
        
    
 
    return obs