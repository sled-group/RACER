# Code Adapted from https://github.com/NVlabs/RVT

import os

from racer.utils.dataset_aug import create_replay
from racer.rvt.utils.peract_utils import (
    CAMERAS,
    VOXEL_SIZES,
)
from racer.rvt.utils.peract_utils import IMAGE_SIZE
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from racer.utils.racer_utils import print0



def get_dataset_v2(
    tasks: list,
    BATCH_SIZE,
    REPLAY_STORAGE_DIR,
    num_workers,
    sample_distribution_mode="task_uniform",
    image_size=IMAGE_SIZE,
):
    replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=image_size,
    )
   
    print0(f"A total of {len(tasks)} tasks to be processed.")
    for task in tasks:  # for each task
        task_replay_storage_folder = f"{REPLAY_STORAGE_DIR}/{task}"
        if os.path.exists(task_replay_storage_folder):
            print0(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            replay_buffer.recover_from_disk(task, task_replay_storage_folder)
        else:
            raise ValueError("Replay dataset not found in the disk. Use preprocess_replay_buffer.py to generate it.")

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()
    test_dataset = None
    val_dataset = None


    return train_dataset, val_dataset, test_dataset
