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
import pickle
import numpy as np
from typing import List

import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation

from racer.peract.helpers.utils import extract_obs
from racer.utils.lang_enc_utils_v2 import LangModelZoo

from racer.rvt.utils.peract_utils import (
    CAMERAS,
    DEPTH_SCALE,
    CAMERA_FRONT,
    CAMERA_LS,
    CAMERA_RS,
    CAMERA_WRIST,
    IMAGE_RGB,
    IMAGE_DEPTH,
)

import os
import pickle
import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor


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




def downsample_array(array, factor):
    if factor == 1:
        return array
    return array[1::factor, 1::factor]

def scale_intrinsics(intrinsics, factor):
    intrinsics[:2, :] /= factor
    return intrinsics

def get_stored_demo_aug(data_path, index, reference_demo):
    episode_path = os.path.join(data_path, str(index))
  
    # low dim pickle file
    with open(os.path.join(episode_path, "obs.pkl"), 'rb') as f:
        _obs = pickle.load(f)

    obs = {}
    for k in _obs.keys():
        if os.path.exists(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), "%s.png" % k)):
            obs[k] = _obs[k]
    
    for key in obs.keys():
        if "dense" in key: 
            continue
        if "expert" in key:
            key_id = int(key.split('_')[0])            
            if "put_groceries_in_cupboard" in data_path: # hot fix for strange data error
                if obs[key].ignore_collisions != reference_demo[key_id].ignore_collisions \
                        and obs[key].ignore_collisions == 0:
                    pass
                else:
                    obs[key].ignore_collisions = reference_demo[key_id].ignore_collisions
            else:
                obs[key].ignore_collisions = reference_demo[key_id].ignore_collisions
            
            obs[key].gripper_open = reference_demo[key_id].gripper_open # use original gripper open just for safe
            obs[key].gripper_pose = reference_demo[key_id].gripper_pose # use original gripper pose just for safe

        obs[key].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), "%s.png" % key)))
        


        obs[key].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), "%s.png" % key)))

        obs[key].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), "%s.png" % key)))
        
        obs[key].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), "%s.png" % key)))

        obs[key].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        near = obs[key].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[key].front_depth = near + obs[key].front_depth * (far - near)

        obs[key].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        near = obs[key].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_LS)]
        obs[key].left_shoulder_depth = near + obs[key].left_shoulder_depth * (far - near)

        obs[key].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        near = obs[key].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_RS)]
        obs[key].right_shoulder_depth = near + obs[key].right_shoulder_depth * (far - near)

        obs[key].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), "%s.png" % key)), DEPTH_SCALE)
        near = obs[key].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[key].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[key].wrist_depth = near + obs[key].wrist_depth * (far - near)

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

    HIGH_LEVEL_GOAL_TEMPLATE = "Task goal: {}."
    LOW_LEVEL_GOAL_TEMPLATE = "{}\nCurrent instruction: {}"
    high_level_task_goal = HIGH_LEVEL_GOAL_TEMPLATE.format(task_goal)

    lang_model_zoo.send_task(text_id="task_goal_embs", text=high_level_task_goal)
    lang_emb_dict = {}
    results = lang_model_zoo.get_results()
    for res in results:
        lang_model_name = res["model"]
        text_id = res["text_id"]
        lang_emb_dict["%s_%s" % (text_id, lang_model_name)] = padding_embs(np.array(res["embeddings"][0], dtype=np.float32))
        lang_emb_dict["%s_%s" % (text_id.replace("_embs", "_len"), lang_model_name)] = np.array([res["token_len"]], dtype=np.int32)

    
    for rich_inst, simp_inst in zip(subgoal_tm1["rich-lang"], subgoal_tm1["simp-lang"]):
        # expert to expert
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(
            deepcopy(obs_tm1),
            CAMERAS,
            t=time_step,
            prev_action=None,
            episode_length=30,
        )
        obs_dict.update(lang_emb_dict)
        obs_dict["ignore_collisions"] = np.array([ignore_collisions], dtype=np.int32)

        # remove depth to save space
        for cname in CAMERAS:
            del obs_dict["%s_depth" % cname]
        
        rich_inst = rich_inst.lower().strip()
        rich_inst = LOW_LEVEL_GOAL_TEMPLATE.format(high_level_task_goal, rich_inst)
        lang_model_zoo.send_task(text_id="rich_inst_embs", text=rich_inst) # query language embeddings through API
        results = lang_model_zoo.get_results()
        for res in results:
            lang_model_name = res["model"]
            text_id = res["text_id"]
            obs_dict["%s_%s" % (text_id, lang_model_name)] = padding_embs(np.array(res["embeddings"][0], dtype=np.float32))
            obs_dict["%s_%s" % (text_id.replace("_embs", "_len"), lang_model_name)] = np.array([
                res["token_len"]], dtype=np.int32)
        
        simp_inst = simp_inst.lower().strip().strip(".")
        simp_inst = LOW_LEVEL_GOAL_TEMPLATE.format(high_level_task_goal, simp_inst)
        lang_model_zoo.send_task(text_id="simp_inst_embs", text=simp_inst)
        results = lang_model_zoo.get_results()
        for res in results:
            lang_model_name = res["model"]
            text_id = res["text_id"]
            obs_dict["%s_%s" % (text_id, lang_model_name)] = padding_embs(np.array(res["embeddings"][0], dtype=np.float32))
            obs_dict["%s_%s" % (text_id.replace("_embs", "_len"), lang_model_name)] = np.array([res["token_len"]], dtype=np.int32)
        
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
            "task_goal": np.array([task_goal], dtype=object),
            "rich_inst": np.array([rich_inst], dtype=object),
            "simp_inst": np.array([simp_inst], dtype=object)
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


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path_train: str,
    reference_data_path_train: str,
    data_path_val: str,
    reference_data_path_val: str,
    lang_model_zoo=None,
):
        
    print("Filling replay in %s ... " % task_replay_storage_folder)
    os.makedirs(task_replay_storage_folder, exist_ok=True)
    for data_path, reference_data_path in [
        (data_path_train, reference_data_path_train),
        (data_path_val, reference_data_path_val),
    ]:          
    
        for ep_name in sorted(os.listdir(data_path), key=lambda x: int(x.split("_")[0])):
            d_idx = int(ep_name.split("_")[0])
            print(f"Filling demo {ep_name} of {data_path}")
            # from original data
            reference_demo = get_stored_demo(data_path=reference_data_path, index=d_idx)
            # from augmented data
            demo = get_stored_demo_aug(data_path=data_path, index=ep_name, reference_demo=reference_demo)            
            # get language goal from disk
            with open(os.path.join(data_path, str(ep_name), "language_description.json"), "r") as f:
                language_description = json.load(f)
            
            # transition
            # type 1: expert to expert, i.e., expert transition
            # type 2: recoverable_failure to intermediate, if has
            # type 3: recoverable_failure to expert, i.e., recovery transition
            
            task_goal = language_description["task_goal"]
            if isinstance(task_goal, list):
                task_goal = task_goal[0] # take easiest one
            expert_step_keys = [s for s in list(language_description["subgoal"].keys()) if "expert" in s]
            expert_step_keys = sorted(expert_step_keys, key=lambda x: int(x.split("_")[0]))
            return_dict = {}
            
            for expert_key_idx in range(1, len(expert_step_keys)):
                key_id = int(expert_step_keys[expert_key_idx].split('_')[0])
                if key_id < 10: continue # skip the first 10 dense steps since it's very unlikely to be an expert step

                current_expert_key = expert_step_keys[expert_key_idx]
                subgoal_tp1 = language_description["subgoal"][current_expert_key]
                obs_tp1 = demo[current_expert_key]

                previous_expert_key = expert_step_keys[expert_key_idx - 1]
                subgoal_tm1 = language_description["subgoal"][previous_expert_key]
                obs_tm1 = demo[previous_expert_key]

                current_subgoal = deepcopy(subgoal_tp1)
                terminal = expert_key_idx == len(expert_step_keys) - 1
                prob = random.random()
                
                # add some randome steps
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

                neglected_last_step = [
                    "insert_onto_square_peg", 
                    "meat_off_grill", 
                    "place_shape_in_shape_sorter",  
                    "put_item_in_drawer",
                    "turn_tap"
                ] # no need to augment failures for the last steps in those tasks, too unnatural

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