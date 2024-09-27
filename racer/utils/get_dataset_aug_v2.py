# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import multiprocessing
import os
import shutil

from racer.utils.dataset_aug_v2 import create_replay, fill_replay
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from racer.utils.lang_enc_utils_v2 import LangModelZoo
from rvt.utils.peract_utils import IMAGE_SIZE


def process_task(
    task:str,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    VAL_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    only_train,
    image_size,
):
    # transform all datasets: train/val/test

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=image_size
    )
    if not only_train:
        val_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_VAL,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            image_size=image_size
        )
    
    lang_model_zoo = LangModelZoo(device)


    EPISODES_FOLDER_TRAIN = f"train/{task}"
    EPISODES_FOLDER_VAL = f"val/{task}"
    data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
    data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)

    DATA_FOLDER_DIR = "data"

    reference_data_path_train = os.path.join(DATA_FOLDER_DIR, f"train/{task}/all_variations/episodes")
    reference_data_path_val = os.path.join(DATA_FOLDER_DIR, f"val/{task}/all_variations/episodes")

    train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
    val_replay_storage_folder = f"{VAL_REPLAY_STORAGE_DIR}/{task}"
    test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

    # if refresh_replay, then remove the existing replay data folder
    if refresh_replay:
        print("[Info] Remove exisitng replay dataset as requested.", flush=True)
        if os.path.exists(train_replay_storage_folder) and os.path.isdir(
            train_replay_storage_folder
        ):
            print(f"removing {train_replay_storage_folder}")
            shutil.rmtree(train_replay_storage_folder)
            print(f"removed {train_replay_storage_folder}")
        if os.path.exists(val_replay_storage_folder) and os.path.isdir(
            val_replay_storage_folder
        ):
            print(f"removing {val_replay_storage_folder}")
            shutil.rmtree(val_replay_storage_folder)
            print(f"removed {val_replay_storage_folder}")
        if os.path.exists(test_replay_storage_folder) and os.path.isdir(
            test_replay_storage_folder
        ):
            print(f"removing {test_replay_storage_folder}")
            shutil.rmtree(test_replay_storage_folder)
            print(f"removed {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            reference_data_path=reference_data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            lang_model_zoo=lang_model_zoo,
            device=device,
            PUT_GROCERIES={1:64, 2:54, 5:74, 6:63, 12:79, 16:62, 20:85, 21:86, 28:65, 30:80, 31:83, 32:58, 35:75, 41:64, 42:58, 98:48, 97:47, 93:54, 91:78, 84:60,83:64, 69:76, 67:68,65:66,62:77,61:75,60:58,55:48,53:70, 52:54}

        )

        if not only_train:
            fill_replay(
                replay=val_replay_buffer,
                task=task,
                task_replay_storage_folder=val_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                reference_data_path=reference_data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                lang_model_zoo=lang_model_zoo,
                device=device,
                PUT_GROCERIES={1:91, 3:90, 6:73, 12:76, 13:44, 15:81,16:81, 17:54, 20:66, 23:80}
            )



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
    image_size=IMAGE_SIZE
):
    
    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(
            process_task,
            [(task,
            BATCH_SIZE_TRAIN,
            BATCH_SIZE_VAL,
            BATCH_SIZE_TEST,
            TRAIN_REPLAY_STORAGE_DIR,
            VAL_REPLAY_STORAGE_DIR,
            TEST_REPLAY_STORAGE_DIR,
            DATA_FOLDER,
            NUM_TRAIN,
            NUM_VAL,
            refresh_replay,
            device,
            only_train,
            image_size) for task in tasks]
        )