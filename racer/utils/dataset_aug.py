# Code Adapted from https://github.com/NVlabs/RVT

from copy import deepcopy
import numpy as np


from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from racer.rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE
from racer.utils.lang_enc_utils_v2 import ALL_MODELS

def get_lang_dim(lang_model_name):
    if lang_model_name == "clip":
        dim = 512
    else:
        dim = 1024
    return dim


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
            ReplayElement("task_goal", (1,), object),  # high-level task goal
            ReplayElement("rich_inst", (1,), object),  # rich instruction
            ReplayElement("simp_inst", (1,), object)   # simple instruction
        ]
    )
    
    for lang_model_name in ALL_MODELS:
        observation_elements.extend([
            ReplayElement(
                "task_goal_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "rich_inst_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement(
                "simp_inst_embs_%s" % lang_model_name,
                (
                    max_token_seq_len,
                    get_lang_dim(lang_model_name),
                ),
                np.float32,
            ),
            ReplayElement("rich_inst_len_%s" % lang_model_name, (1,), np.int32),
            ReplayElement("simp_inst_len_%s" % lang_model_name, (1,), np.int32),
            ReplayElement("task_goal_len_%s" % lang_model_name, (1,), np.int32),
        ])
            

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