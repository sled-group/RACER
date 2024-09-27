# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import shutil

from rvt.utils.dataset_aug import create_replay, fill_replay
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
from rvt.utils.lang_enc_utils import LangModelZoo
from rvt.utils.peract_utils import IMAGE_SIZE


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
    # transform all datasets: train/val/test

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=image_size,
        other_lang_path=other_lang_path
    )
    if not only_train:
        test_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            image_size=image_size,
            other_lang_path=other_lang_path
        )
        val_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_VAL,
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

    print(f" A total of {len(tasks)} tasks to be processed.")
    for task in tasks:  # for each task
         # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}"
        EPISODES_FOLDER_VAL = f"val/{task}"
        EPISODES_FOLDER_TEST = f"test/{task}"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        data_path_test = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TEST)

        DATA_FOLDER_DIR = os.path.dirname(DATA_FOLDER)

        reference_data_path_train = os.path.join(DATA_FOLDER_DIR, f"train/{task}/all_variations/episodes")
        reference_data_path_val = os.path.join(DATA_FOLDER_DIR, f"val/{task}/all_variations/episodes")
        reference_data_path_test = os.path.join(DATA_FOLDER_DIR, f"test/{task}/all_variations/episodes")

        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        val_replay_storage_folder = f"{VAL_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(val_replay_storage_folder) and os.path.isdir(
                val_replay_storage_folder
            ):
                shutil.rmtree(val_replay_storage_folder)
                print(f"remove {val_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

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
            only_data_check=only_data_check,
            PUT_GROCERIES={1:64, 2:54, 5:74, 6:63, 12:79, 16:62, 20:85, 21:86, 28:65, 30:80, 31:83, 32:58, 35:75, 41:64, 42:58, 98:48, 97:47, 93:54, 91:78, 84:60,83:64, 69:76, 67:68,65:66,62:77,61:75,60:58,55:48,53:70, 52:54}

        )

        if not only_train:
            # # print("----- Test Buffer -----")
            # fill_replay(
            #     replay=test_replay_buffer,
            #     task=task,
            #     task_replay_storage_folder=test_replay_storage_folder,
            #     start_idx=0,
            #     num_demos=NUM_TEST,
            #     demo_augmentation=True,
            #     demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            #     cameras=CAMERAS,
            #     rlbench_scene_bounds=SCENE_BOUNDS,
            #     voxel_sizes=VOXEL_SIZES,
            #     rotation_resolution=ROTATION_RESOLUTION,
            #     crop_augmentation=False,
            #     data_path=data_path_test,
            #     reference_data_path=reference_data_path_test,
            #     episode_folder=EPISODE_FOLDER,
            #     variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            #     lang_model_zoo=lang_model_zoo,
            #     device=device,
            #     only_data_check=only_data_check,
            # )

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
                only_data_check=only_data_check,
                PUT_GROCERIES={1:91, 3:90, 6:73, 12:76, 13:44, 15:81,16:81, 17:54, 20:66, 23:80}
            )
    
    if only_data_check:
        return

    if only_data_gen:
        lang_model_zoo.unmake_models()
        return None

    else:
        # wrap buffer with PyTorch dataset and make iterator
        train_wrapped_replay = PyTorchReplayBuffer(
            train_replay_buffer,
            sample_mode="random",
            num_workers=num_workers,
            sample_distribution_mode=sample_distribution_mode,
        )
        train_dataset = train_wrapped_replay.dataset()

        if only_train:
            test_dataset = None
            val_dataset = None
        else:
            test_wrapped_replay = PyTorchReplayBuffer(
                test_replay_buffer,
                sample_mode="enumerate",
                num_workers=num_workers,
            )
            test_dataset = test_wrapped_replay.dataset()

            val_wrapped_replay = PyTorchReplayBuffer(
                val_replay_buffer,
                sample_mode="enumerate",
                num_workers=num_workers,
            )
            val_dataset = val_wrapped_replay.dataset()


        return train_dataset, val_dataset, test_dataset
