# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import random
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from peract.helpers.demo_loading_utils import keypoint_discovery
from peract.helpers.utils import extract_obs
from rvt.utils.lang_enc_utils import LangModel, LangModelZoo

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


ALL_MODELS = ["clip", 
        "t5-3b", "t5-11b", 
        "bart-large", "roberta-large", 
        "llama3"
        ]

def get_lang_dim(lang_model_name):
    if lang_model_name == "clip":
        DIM = 512
    elif lang_model_name == "llama3":
        DIM = 4096
    else:
        DIM = 1024
    return DIM

def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization, 
        ]
    )
    
    for lang_model_name in ALL_MODELS:
        observation_elements.extend([
            ReplayElement(
                "lang_goal_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_len_%s" % lang_model_name, (1,), np.int32
            )  # length of language tokens
        ]
    )
                

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


# add individual data points to a replay
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    lang_model_zoo: LangModelZoo = None,
):
    prev_action = None
    obs = inital_obs
    
    # # set a small ratio to repeat similar samples
    # if len(episode_keypoints) == 1:
    #     dont_repeat = False
    # else:
    #     dont_repeat = random.random() > 2 / (len(episode_keypoints)-1)


    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        (
            trans_indicies,
            rot_grip_indicies,
            _,
            action,
            _,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k-next_keypoint_idx,
            prev_action=prev_action,
            episode_length=30,
        )
        ignore_collisions = obs_tm1.ignore_collisions or obs_tp1.ignore_collisions
        obs_dict["ignore_collisions"] = np.array([ignore_collisions], dtype=np.int32)
        
        task_goal = "task goal: " + description + "."
        all_lang_embds = lang_model_zoo.encode(task_goal)
        all_lang_lens = lang_model_zoo.token_len(task_goal)
        for lang_model_name in ALL_MODELS:
            obs_dict["lang_goal_embs_%s" % lang_model_name] = all_lang_embds[lang_model_name][0]
            obs_dict["lang_len_%s" % lang_model_name] = np.array([all_lang_lens[lang_model_name]], dtype=np.int32) 
        

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

        # if dont_repeat:
        #     break

    # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    
    for lang_model_name in ALL_MODELS:
        obs_dict_tp1["lang_goal_embs_%s" % lang_model_name] = all_lang_embds[lang_model_name][0]
        obs_dict_tp1["lang_len_%s" % lang_model_name] = np.array([all_lang_lens[lang_model_name]], dtype=np.int32)

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


FilteredEpisodes= {
  "reach_and_drag": [84],
  "turn_tap": [69, 74, 77, 86],
  "open_drawer": [37],
  "put_groceries_in_cupboard": [36, 51, 54, 58, 82, 89, 90],
  "place_shape_in_shape_sorter": [30, 41, 44, 64, 83, 87],
  "close_jar": [47],
  "stack_blocks": [24, 74],
  "place_cups": [10, 12, 19, 21, 30, 34, 38, 44, 56, 59, 60, 61, 65, 70, 80, 82, 83, 88, 92, 97, 98], # TODO 
  "light_bulb_in": [1, 29],
  "stack_cups": [90]
}


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    lang_model_zoo=None,
    device="cpu",
):
    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            print("\t saving to disk: %s" % task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        if lang_model_zoo is None:
            assert False, "please choose the right replay buffer for model training, or use data_gen.py to generate data first"
        print("Filling replay in %s ... " % task_replay_storage_folder)
        for d_idx in tqdm(range(start_idx, start_idx + num_demos)):
            # filter failed episodes
            if d_idx in FilteredEpisodes.get(task, []) and "replay_train" in task_replay_storage_folder:
                print(f"Failed episode, skipping demo {task} episode {d_idx}")
                continue

            # print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)
            episode_keypoints = [k for k in episode_keypoints if k > 5]
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = random.choice(descs)
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay(
                    replay=replay,
                    task=task,
                    task_replay_storage_folder=task_replay_storage_folder,
                    episode_idx=d_idx,
                    sample_frame=i,
                    inital_obs=obs,
                    demo=demo,
                    episode_keypoints=episode_keypoints,
                    cameras=cameras,
                    rlbench_scene_bounds=rlbench_scene_bounds,
                    voxel_sizes=voxel_sizes,
                    rotation_resolution=rotation_resolution,
                    crop_augmentation=crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    lang_model_zoo=lang_model_zoo,
                )

        # # save TERMINAL info in replay_info.npy
        # task_idx = replay._task_index[task]
        # with open(
        #     os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        # ) as fp:
        #     np.save(
        #         fp,
        #         replay._store["terminal"][
        #             replay._task_replay_start_index[
        #                 task_idx
        #             ] : replay._task_replay_start_index[task_idx]
        #             + replay._task_add_count[task_idx].value
        #         ],
        #     )

        # print("Replay filled with demos.")
