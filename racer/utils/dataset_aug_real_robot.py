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

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation

from racer.utils.real_robot_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS, MAX_POINTS
from racer.peract.helpers.demo_loading_utils import keypoint_discovery
from racer.utils.lang_enc_utils_v2 import LangModelZoo, ALL_MODELS


FAILURE_LABELS = {
    "start": 0,
    "recoverable_failure": 1,
    "catastrophic_failure": 2,
    "success": 3,
    "ongoing": 4,
}



def extract_obs(obs: dict,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False,
                episode_length: int = 10):

    obs_dict = {}    
    gripper_joint_positions = np.clip(obs["gripper_joint_positions"], 0., 0.04)
    robot_state = np.array([obs["gripper_open"], *gripper_joint_positions], dtype=np.float32)  
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    number_points = obs["merged_pc"].shape[0]
    obs_dict["merged_pc"] = np.concatenate(
        [obs["merged_pc"], np.zeros((MAX_POINTS - number_points, 6), dtype=np.float32)], axis=0)
    obs_dict["merged_pc_number"] = np.array([number_points], dtype=np.int32)

    return obs_dict



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
    other_lang_path=None,
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
    observation_elements.append(
        ObservationElement("merged_pc", (MAX_POINTS,6), np.float32)
    )
    observation_elements.append(
        ObservationElement("merged_pc_number", (1,), np.int32)
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
                "fine_gpt_lang_goal-level1", (1,), object
            ),
            ReplayElement(
                "fine_gpt_lang_goal-level2", (1,), object
            ),
        ]
    )

    for lang_model_name in ["t5-11b"]:
        observation_elements.extend([
            ReplayElement(
                "fine_gpt_lang_goal_embs_%s-level1" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement(
                "fine_gpt_lang_goal_embs_%s-level2" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement(
                "fine_gpt_lang_len_%s-level1" % lang_model_name, (1,), np.int32
            ),
            ReplayElement(
                "fine_gpt_lang_len_%s-level2" % lang_model_name, (1,), np.int32
            ),
        ]
    )
    
    for lang_model_name in  [
        "clip", 
        "t5-3b", 
        "t5-11b", 
        "bart-large", "roberta-large", 
        "llama3"
        ]:
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
            other_lang_path=other_lang_path,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs: dict,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
):
    quat = utils.normalize_quaternion(obs["gripper_pose"][3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs["gripper_pose"][:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = 0
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs["gripper_pose"][:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs["gripper_open"])
    rot_and_grip_indicies.extend([int(obs["gripper_open"])])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs["gripper_pose"], np.array([grip])]),
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

    HIGH_LEVEL_GOAL_TEMPLATE = "Task goal: {}."
    LOW_LEVEL_GOAL_TEMPLATE = "{}\nCurrent instruction: {}"
    high_level_task_goal = HIGH_LEVEL_GOAL_TEMPLATE.format(task_goal)

    assert isinstance(subgoal_tm1["gpt-lang"], list), "please check the data"


    lang_model_zoo.send_task(text_id="lang_goal_embs", text=high_level_task_goal)

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
            "gripper_pose": obs_tp1["gripper_pose"],
            "lang_goal": np.array([task_goal], dtype=object),
            "fine_gpt_lang_goal": np.array([fine_grained_gpt_lang], dtype=object),
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
    start_idx: int,
    num_demos: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    lang_model_zoo=None,
    device="cpu"
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

            # get language goal from disk
            with open(os.path.join(data_path, str(d_idx), "language_description.json"), "r") as f:
                language_description = json.load(f)
                
            with open(os.path.join(data_path, str(d_idx), "obs.pkl"), "rb") as f:
                demo = pickle.load(f)

            
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
                current_expert_key = expert_step_keys[expert_key_idx]
                subgoal_tp1 = language_description["subgoal"][current_expert_key]
                obs_tp1 = demo[current_expert_key]

                previous_expert_key = expert_step_keys[expert_key_idx - 1]
                subgoal_tm1 = language_description["subgoal"][previous_expert_key]
                obs_tm1 = demo[previous_expert_key]

                current_subgoal = deepcopy(subgoal_tp1)

                terminal = expert_key_idx == len(expert_step_keys) - 1

               
                time_step = expert_key_idx

                # previous expert to current expert
                if task == "open_drawer" and d_idx in [8, 9, 10, 11] and expert_key_idx == 3:
                    pass
                else:    
                    print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])   
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


                # if "augmentation" in current_subgoal and current_subgoal["augmentation"]:
                #     for k_mistake, v_mistake in current_subgoal["augmentation"].items():
                #         if "perturb" not in k_mistake: continue                        
                #         obs_tm1 = demo[k_mistake]
                #         subgoal_tm1 = v_mistake

                #         if v_mistake["correction"]:
                #             # current mistake to intermediate correction
                #             k_correct = list(v_mistake["correction"].keys())[0]
                #             v_correct = v_mistake["correction"][k_correct]
                #             obs_tp1 = demo[k_correct]
                #             subgoal_tp1 = v_correct
                #         else:
                #             # mistake to current expert
                #             pass

                #         print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])                            
                #         return_dict = _add_keypoints_to_replay(
                #             replay,
                #             task,
                #             task_replay_storage_folder,
                #             terminal,
                #             time_step,
                #             obs_tp1,
                #             subgoal_tp1,
                #             obs_tm1,
                #             subgoal_tm1,
                #             rlbench_scene_bounds,
                #             voxel_sizes,
                #             rotation_resolution,
                #             crop_augmentation,
                #             task_goal,
                #             lang_model_zoo,
                #             is_expert_step=False,
                #             keypoint_idx=-1,
                #             episode_idx=d_idx,
                #         )

                #         if v_mistake["correction"]:
                #             # intermediate to current expert
                #             k_correct = list(v_mistake["correction"].keys())[0]
                #             obs_tm1 = demo[k_correct]
                #             subgoal_tm1 = v_mistake["correction"][k_correct]
                #             obs_tp1 = demo[current_expert_key]
                #             subgoal_tp1 = language_description["subgoal"][current_expert_key]


                #             print(subgoal_tm1["idx"], "->", subgoal_tp1["idx"])   
                #             return_dict = _add_keypoints_to_replay(
                #                 replay,
                #                 task,
                #                 task_replay_storage_folder,
                #                 terminal,
                #                 time_step+1,
                #                 obs_tp1,
                #                 subgoal_tp1,
                #                 obs_tm1,
                #                 subgoal_tm1,
                #                 rlbench_scene_bounds,
                #                 voxel_sizes,
                #                 rotation_resolution,
                #                 crop_augmentation,
                #                 task_goal,
                #                 lang_model_zoo,
                #                 is_expert_step=False,
                #                 keypoint_idx=-2,
                #                 episode_idx=d_idx,
                #             )

            delete_keys = ["demo", "keypoint_idx", "episode_idx", "keypoint_frame", "next_keypoint_frame", "sample_frame"]
            if any(k not in return_dict for k in delete_keys): # no any replay buffer has been added for this episode
                print("skip episode", d_idx, "since no perutation has been added")
                continue
            for k in delete_keys: return_dict.pop(k)
            replay.add_final(task, task_replay_storage_folder, **return_dict)
        print("Replay filled with demos.")