# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
from copy import deepcopy
import json
import os
import random
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from rvt.utils.peract_utils import get_stored_demo_aug
from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs
from racer.utils.lang_enc_utils_v2 import LangModelZoo, ALL_MODELS

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

FAILURE_LABELS = {
    "start": 0,
    "recoverable_failure": 1,
    "catastrophic_failure": 2,
    "success": 3,
    "ongoing": 4,
}



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
    image_size=IMAGE_SIZE,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    failure_cls_size = len(FAILURE_LABELS)

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
                    image_size,
                    image_size,
                ),
                np.float32,
            )
        )
        # observation_elements.append(
        #     ObservationElement(
        #         "%s_depth" % cname,
        #         (
        #             1,
        #             image_size,
        #             image_size,
        #         ),
        #         np.float32,
        #     )
        # )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    image_size,
                    image_size,
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
            ReplayElement(
                "fine_gpt_lang_goal", (1,), object
            ), 
            ReplayElement(
                "fine_heuristic_lang_goal", (1,), object
            ), 
            ReplayElement(
                "failure_labels", (1,), np.int32
            ),
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
            ),  # length of language tokens
            ReplayElement(
                "fine_gpt_lang_goal_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement(
                "fine_gpt_lang_len_%s" % lang_model_name, (1,), np.int32
            ),  # length of language tokens for fine-grained language instructions
            ReplayElement(
                "fine_heuristic_lang_goal_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement(
                "fine_heuristic_lang_len_%s" % lang_model_name, (1,), np.int32
            ),  # length of language tokens for fine-grained language instructions
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
    obs: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
):
    quat = utils.normalize_quaternion(obs.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs.gripper_open)
    rot_and_grip_indicies.extend([int(obs.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )

def padding_embs(embs, max_len=77):
    shape = embs.shape
    if shape[0] > max_len:
        return embs[:max_len, :]
    else:
        padding = np.zeros((max_len - shape[0], shape[1]), dtype=np.float32)
        return np.concatenate([embs, padding], axis=0)



def _add_keypoints_to_replay(
    replay: UniformReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    terminal: bool,
    time_step: int,
    obs_tp1: Observation,
    subgoal_tp1: dict,
    obs_tm1: Observation,
    subgoal_tm1: dict,
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    task_goal: str,
    lang_model_zoo: LangModelZoo,
    is_expert_step=False,
    keypoint_idx=-1,
    episode_idx=-1,
):    

    (
        trans_indicies,
        rot_grip_indicies,
        ignore_collisions,
        action,
        _,
    ) = _get_action(
        obs_tp1,
        rlbench_scene_bounds,
        voxel_sizes,
        rotation_resolution,
    )

    # print("trans_indicies", trans_indicies)
    # print("rot_grip_indicies", rot_grip_indicies)
    # print("ignore_collisions", ignore_collisions)

    failure_label = subgoal_tm1["label"]
    fine_grained_heuristic_lang = subgoal_tm1["heuristic-lang"].lower().strip()
    HIGH_LEVEL_GOAL_TEMPLATE = "Task goal: {}."
    LOW_LEVEL_GOAL_TEMPLATE = "{}\nCurrent instruction: {}"
    high_level_task_goal = HIGH_LEVEL_GOAL_TEMPLATE.format(task_goal)
    fine_grained_heuristic_lang = LOW_LEVEL_GOAL_TEMPLATE.format(high_level_task_goal, fine_grained_heuristic_lang)

    assert isinstance(subgoal_tm1["gpt-lang"], list), "please check the data"


    lang_model_zoo.send_task(text_id="lang_goal_embs", text=high_level_task_goal)
    lang_model_zoo.send_task(text_id="fine_heuristic_lang_goal_embs", text=fine_grained_heuristic_lang)

    lang_emb_dict = {}
    results = lang_model_zoo.get_results()
    for res in results:
        lang_model_name = res["model"]
        text_id = res["text_id"]
        lang_emb_dict["%s_%s" % (text_id, lang_model_name)] = padding_embs(np.array(res["embeddings"][0], dtype=np.float32))
        lang_emb_dict["%s_%s" % (text_id.replace("goal_embs", "len"), lang_model_name)] = np.array([res["token_len"]], dtype=np.int32)

    
    for fine_grained_gpt_lang in subgoal_tm1["gpt-lang"]:
        fine_grained_gpt_lang = fine_grained_gpt_lang.lower().strip()
        fine_grained_gpt_lang = LOW_LEVEL_GOAL_TEMPLATE.format(high_level_task_goal, fine_grained_gpt_lang)

        # query language embeddings through API
        lang_model_zoo.send_task(text_id="fine_gpt_lang_goal_embs", text=fine_grained_gpt_lang)

        # expert to expert
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(
            deepcopy(obs_tm1),
            CAMERAS,
            t=time_step,
            prev_action=None,
            episode_length=30,
        )
        obs_dict["ignore_collisions"] = np.array([ignore_collisions], dtype=np.int32)
        obs_dict["failure_labels"] = np.array([FAILURE_LABELS[failure_label]], dtype=np.int32)

        # remove depth to save space
        for cname in CAMERAS:
            del obs_dict["%s_depth" % cname]


        # get language embeddings results        
        results = lang_model_zoo.get_results()
        for res in results:
            lang_model_name = res["model"]
            text_id = res["text_id"]
            obs_dict["%s_%s" % (text_id, lang_model_name)] = padding_embs(np.array(res["embeddings"][0], dtype=np.float32))
            obs_dict["%s_%s" % (text_id.replace("goal_embs", "len"), lang_model_name)] = np.array([res["token_len"]], dtype=np.int32)
        
        obs_dict.update(lang_emb_dict)

        others = {
            "demo": is_expert_step,
            "keypoint_idx": keypoint_idx,
            "episode_idx": episode_idx,
            "keypoint_frame": 0,
            "next_keypoint_frame": 0,
            "sample_frame": 0,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([task_goal], dtype=object),
            "fine_gpt_lang_goal": np.array([fine_grained_gpt_lang], dtype=object),
            "fine_heuristic_lang_goal": np.array([fine_grained_heuristic_lang], dtype=object),
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

    return others


# PUT_GROCERIES= {1:64, 2:54, 5:74, 6:63, 12:79, 16:62, 20:85, 21:86, 28:65, 30:80, 31:83, 32:58, 35:75, 41:64, 42:58, 
# 98:48, 97:47, 93:54, 91:78, 84:60,83:64, 69:76, 67:68,65:66,62:77,61:75,60:58,55:48,53:70, 52:54} # training set
# # PUT_GROCERIES = {1:91, 3:90, 6:73, 12:76, 13:44, 15:81,16:81, 17:54, 20:66, 23:80} # val set

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
    reference_data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    lang_model_zoo=None,
    device="cpu",
    PUT_GROCERIES=None
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
        for d_idx in range(start_idx, start_idx + num_demos):
            episode_path = os.path.join(data_path, str(d_idx))
            # expert failures
            if not os.path.exists(episode_path):
                continue
            # missing language description json
            if not os.path.exists(os.path.join(data_path, str(d_idx), "language_description.json")):
                continue

            print(f"Filling demo {d_idx} of {task_replay_storage_folder}")
            # from original data
            reference_demo = get_stored_demo(data_path=reference_data_path, index=d_idx)
            # from augmented data
            demo = get_stored_demo_aug(data_path=data_path, index=d_idx, reference_demo=reference_demo)

            # get language goal from disk
            with open(os.path.join(data_path, str(d_idx), "language_description.json"), "r") as f:
                language_description = json.load(f)

            # TODO: compare language_description key and demo key
            
            # transition
            # type 1: expert to expert
            # type 2: recoverable_failure to intermediate, if has
            # type 3: recoverable_failure to expert

            task_goal = language_description["task_goal"]
            if isinstance(task_goal, list):
                task_goal = task_goal[0] # take easiest one
            expert_step_keys = [s for s in list(language_description["subgoal"].keys()) if "expert" in s]
            expert_step_keys = sorted(expert_step_keys, key=lambda x: int(x.split("_")[0]))

            return_dict = {}
            
            for expert_key_idx in range(1, len(expert_step_keys)):
                key_id = int(expert_step_keys[expert_key_idx].split('_')[0])
                if key_id < 10: continue

                current_expert_key = expert_step_keys[expert_key_idx]
                subgoal_tp1 = language_description["subgoal"][current_expert_key]
                obs_tp1 = demo[current_expert_key]

                previous_expert_key = expert_step_keys[expert_key_idx - 1]
                subgoal_tm1 = language_description["subgoal"][previous_expert_key]
                obs_tm1 = demo[previous_expert_key]

                current_subgoal = deepcopy(subgoal_tp1)

                terminal = expert_key_idx == len(expert_step_keys) - 1

                if "put_groceries_in_cupboard" in data_path and not terminal and PUT_GROCERIES: # fix
                    if d_idx in PUT_GROCERIES:
                        if key_id < PUT_GROCERIES[d_idx]: 
                            continue # skip uneccessary steps
                        if key_id == PUT_GROCERIES[d_idx]:
                            demo[current_expert_key].ignore_collisions = np.array(0.)
                            subgoal_tm1 = language_description["subgoal"]["0_expert"]
                            obs_tm1 = demo["0_expert"]

                        align_step = PUT_GROCERIES[d_idx]
                    else:
                        align_step = int(expert_step_keys[1].split('_')[0])

                    # fix gripper_open
                    if key_id > align_step:
                        if demo[current_expert_key].gripper_open == 1.:
                            print(key_id, "this is a mistake, please check the data")
                            demo[current_expert_key].gripper_open = 0.

                prob = random.random()
                if prob < 0.7 or expert_key_idx == 1:
                    time_step = expert_key_idx
                elif 0.7<=prob <0.85:
                    time_step = expert_key_idx + 1
                else:
                    time_step = expert_key_idx + 2

                # previous expert to current expert
                # print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])   
                return_dict = _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    terminal,
                    time_step-1,
                    obs_tp1,
                    subgoal_tp1,
                    obs_tm1,
                    subgoal_tm1,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    task_goal,
                    lang_model_zoo,
                    is_expert_step=True,
                    keypoint_idx=key_id,
                    episode_idx=d_idx,
                )

                neglected_last_step = ["insert_onto_square_peg", "meat_off_grill", "place_shape_in_shape_sorter",  "put_item_in_drawer", "turn_tap"]

                if "augmentation" in current_subgoal and current_subgoal["augmentation"]:
                    if "put_groceries_in_cupboard" in  data_path and expert_key_idx >= len(expert_step_keys) - 2:
                        continue
                    if any([k in data_path for k in neglected_last_step]) and expert_key_idx >= len(expert_step_keys) - 1:
                        continue
                    for k_mistake, v_mistake in current_subgoal["augmentation"].items():
                        if "perturb" not in k_mistake: continue                        
                        obs_tm1 = demo[k_mistake]
                        subgoal_tm1 = v_mistake

                        # compare the perturbation and the expert difference
                        trans_indicies_perturb, rot_grip_indicies_perturb, _, _, _, = _get_action(
                                deepcopy(demo[k_mistake]), rlbench_scene_bounds, voxel_sizes, rotation_resolution)
                        
                        trans_indicies, rot_grip_indicies, _, _, _, = _get_action(
                                deepcopy(demo[current_expert_key]), rlbench_scene_bounds, voxel_sizes, rotation_resolution)
                        trans_indicies_perturb = np.array(trans_indicies_perturb)
                        rot_grip_indicies_perturb = np.array(rot_grip_indicies_perturb)
                        trans_indicies = np.array(trans_indicies)
                        rot_grip_indicies = np.array(rot_grip_indicies)
                        
                        if np.linalg.norm(trans_indicies-trans_indicies_perturb)<=1 and np.linalg.norm(rot_grip_indicies[:3]-rot_grip_indicies_perturb[:3])<=1 and rot_grip_indicies[3]==rot_grip_indicies_perturb[3]:
                            print(f"{k_mistake} perturbation is too small, skip...")
                            continue
                        if np.linalg.norm(trans_indicies-trans_indicies_perturb)<2 and task_goal == "put the cylinder in the shape sorter":
                            print(f"{k_mistake} perturbation not valid for cylinder, skip...")
                            continue

                        if v_mistake["correction"]:
                            # current mistake to intermediate correction
                            k_correct = list(v_mistake["correction"].keys())[0]
                            v_correct = v_mistake["correction"][k_correct]
                            obs_tp1 = demo[k_correct]
                            subgoal_tp1 = v_correct
                        else:
                            # mistake to current expert
                            pass

                        # print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])                            
                        return_dict = _add_keypoints_to_replay(
                            replay,
                            task,
                            task_replay_storage_folder,
                            terminal,
                            time_step,
                            obs_tp1,
                            subgoal_tp1,
                            obs_tm1,
                            subgoal_tm1,
                            rlbench_scene_bounds,
                            voxel_sizes,
                            rotation_resolution,
                            crop_augmentation,
                            task_goal,
                            lang_model_zoo,
                            is_expert_step=False,
                            keypoint_idx=-1,
                            episode_idx=d_idx,
                        )

                        if v_mistake["correction"]:
                            # intermediate to current expert
                            k_correct = list(v_mistake["correction"].keys())[0]
                            obs_tm1 = demo[k_correct]
                            subgoal_tm1 = v_mistake["correction"][k_correct]
                            obs_tp1 = demo[current_expert_key]
                            subgoal_tp1 = language_description["subgoal"][current_expert_key]


                            # print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])   
                            return_dict = _add_keypoints_to_replay(
                                replay,
                                task,
                                task_replay_storage_folder,
                                terminal,
                                time_step+1,
                                obs_tp1,
                                subgoal_tp1,
                                obs_tm1,
                                subgoal_tm1,
                                rlbench_scene_bounds,
                                voxel_sizes,
                                rotation_resolution,
                                crop_augmentation,
                                task_goal,
                                lang_model_zoo,
                                is_expert_step=False,
                                keypoint_idx=-2,
                                episode_idx=d_idx,
                            )

            delete_keys = ["demo", "keypoint_idx", "episode_idx", "keypoint_frame", "next_keypoint_frame", "sample_frame"]
            if any(k not in return_dict for k in delete_keys): # no any replay buffer has been added for this episode
                print("skip episode", d_idx, "since no perutation has been added")
                continue
            for k in delete_keys: return_dict.pop(k)
            replay.add_final(task, task_replay_storage_folder, **return_dict)
        print("Replay filled with demos.")