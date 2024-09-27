# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import multiprocessing
import os
import shutil

from racer.utils.dataset_aug_real_robot import create_replay, fill_replay
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from racer.utils.lang_enc_utils_v2 import LangModelZoo
from racer.utils.real_robot_utils import *



def get_dataset_v2(
    tasks: list,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    VAL_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    NUM_TEST,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="task_uniform",
    only_data_gen=True, # for data generation only
    only_data_check=False, # for data checking only
    image_size=IMAGE_SIZE,
    other_lang_path=None
):
    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=image_size,
        other_lang_path=other_lang_path

    )
    
    if only_data_gen:
        lang_model_zoo = LangModelZoo(device)
        lang_model_zoo.make_models()
    else:
        lang_model_zoo = None

    for task in tasks:
        EPISODES_FOLDER_TRAIN = f"train/{task}"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                print(f"removing {train_replay_storage_folder}")
                shutil.rmtree(train_replay_storage_folder)
                print(f"removed {train_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            lang_model_zoo=lang_model_zoo,
            device=device,

        )
    
    if only_data_gen:
        lang_model_zoo.unmake_models()
        return None

    else:
        train_wrapped_replay = PyTorchReplayBuffer(
            train_replay_buffer,
            sample_mode="random",
            num_workers=num_workers,
            sample_distribution_mode=sample_distribution_mode,
        )
        train_dataset = train_wrapped_replay.dataset()
        return train_dataset

